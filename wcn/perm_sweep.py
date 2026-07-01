# SPDX-License-Identifier: MIT
"""Determine the exact ME->WCN kernel-offset reindexing empirically.

verify_conv.py pinned WCN's conv ordering (meshgrid-'ij', z-fastest, centered,
cross-correlation). The original FCGF weights were trained under MinkowskiEngine's
ordering. The map between them is some element of the octahedral group acting on
the 3x3x3 offset grid (axis permutation + optional per-axis flip). We try all 48,
convert the pretrained weights accordingly, and score each by a SHARP feature
metric: mean inlier-ratio of nearest-neighbor feature matches under the GT pose
(no random rotation, tight threshold). The correct reindexing maximizes it; wrong
ones scramble the learned kernels and collapse the score.
"""

import itertools
import os
import sys

import numpy as np
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_me_to_wcn import me_key_to_wcn  # noqa: E402
from warpconvnet.geometry.types.voxels import Voxels  # noqa: E402
from warpconvnet.models.fcgf import ResUNetBN2C


@torch.no_grad()
def extract_capped(model, xyz, voxel_size, device, cap, seed):
    """Voxelize then subsample to EXACTLY `cap` voxels (constant N -> autotune once).
    Returns (points[cap,3], features[cap,C])."""
    coords = np.floor(xyz / voxel_size).astype(np.int32)
    _, sel = np.unique(coords, axis=0, return_index=True)
    sel = np.sort(sel)
    if len(sel) < cap:
        return None
    rng = np.random.RandomState(seed)
    keep = np.sort(rng.choice(len(sel), cap, replace=False))
    sel = sel[keep]
    c = torch.from_numpy(coords[sel])
    f = torch.ones((cap, 1), dtype=torch.float32)
    out = model(Voxels([c], [f]).to(device))
    return xyz[sel], out.feature_tensor.detach().cpu().numpy()


def wcn_offsets(k=3):
    r = [torch.arange(k) for _ in range(3)]
    g = torch.meshgrid(*r, indexing="ij")
    return torch.stack([x.flatten() for x in g], dim=1) - (k - 1) // 2  # [K^3,3]


def octahedral_transforms(group="full"):
    """Yield (name, 3x3 int signed-permutation matrix).

    group="perms": the 6 pure axis permutations (raveling-order candidates).
    group="full" : all 48 octahedral symmetries (axis perms x per-axis flips).
    """
    sign_sets = [(1, 1, 1)] if group == "perms" else list(itertools.product((1, -1), repeat=3))
    for perm in itertools.permutations(range(3)):
        for signs in sign_sets:
            M = np.zeros((3, 3), dtype=np.int64)
            for i, p in enumerate(perm):
                M[i, p] = signs[i]
            yield (f"perm{perm}sign{signs}", M)


def kernel_reindex(k, M):
    """Weight-index permutation `idx` s.t. new_W[i] = old_W[idx[i]] realizes the
    offset transform o -> M o (new kernel offset i takes the old weight whose
    offset maps to it)."""
    offs = wcn_offsets(k).numpy()  # [K^3,3]
    key = {tuple(o): i for i, o in enumerate(offs)}
    idx = np.empty(len(offs), dtype=np.int64)
    for i, o in enumerate(offs):
        mo = tuple((M @ o).tolist())
        idx[i] = key[mo]
    return idx


def build_wcn_sd(me_sd, k1, reindex_k1, reindex_k3):
    """ME state_dict -> WCN state_dict, applying the given index perms to conv kernels."""
    new = {}
    for mk, mv in me_sd.items():
        wk = me_key_to_wcn(mk)
        if wk is None:
            continue
        v = mv
        if mk.endswith(".kernel"):
            if v.dim() == 2:  # k=1 conv -> [1,in,out], no reindex
                v = v.reshape(1, v.shape[0], v.shape[1])
            else:
                K = v.shape[0]
                ridx = reindex_k1 if K == k1 ** 3 else reindex_k3
                v = v[ridx].contiguous()
        elif mk == "final.bias":
            v = v.reshape(-1)
        new[wk] = v.clone()
    return new


@torch.no_grad()
def mean_inlier_ratio(model, pairs, voxel_size, device, tau, cap):
    ratios = []
    for xyz0, xyz1, T in pairs:
        r0 = extract_capped(model, xyz0, voxel_size, device, cap, seed=1)
        r1 = extract_capped(model, xyz1, voxel_size, device, cap, seed=2)
        if r0 is None or r1 is None:
            continue
        (d0, f0), (d1, f1) = r0, r1
        nn = cKDTree(f1).query(f0, k=1)[1]
        d0t = d0 @ T[:3, :3].T + T[:3, 3]
        dist = np.linalg.norm(d0t - d1[nn], axis=1)
        ratios.append(float(np.mean(dist < tau)))
    return float(np.mean(ratios)) if ratios else 0.0


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--me_ckpt", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--config_dir", required=True)
    p.add_argument("--voxel_size", type=float, default=0.025)
    p.add_argument("--tau", type=float, default=0.05)
    p.add_argument("--n_pairs", type=int, default=8)
    p.add_argument("--cap", type=int, default=5000, help="fixed #voxels/fragment (constant N -> autotune once)")
    p.add_argument("--group", default="perms", choices=["perms", "full"])
    args = p.parse_args()
    device = "cuda"

    ck = torch.load(args.me_ckpt, map_location="cpu", weights_only=False)
    me_sd = ck["state_dict"]
    cfg = ck.get("config")
    conv1_ks = getattr(cfg, "conv1_kernel_size", 7)
    n_out = getattr(cfg, "model_n_out", 32)

    # Build a fixed set of GT-aligned val pairs WITHOUT random rotation (GT=identity
    # for these pre-aligned fragments).
    import glob
    scenes = open(os.path.join(args.config_dir, "val_3dmatch.txt")).read().split()
    files = []
    for s in scenes:
        for txt in glob.glob(os.path.join(args.root, s + "*0.30.txt")):
            for line in open(txt):
                a, b = line.split()[:2]
                files.append((a, b))
    rng = np.random.RandomState(0)
    sel = rng.choice(len(files), min(len(files), args.n_pairs), replace=False)
    pairs = []
    for i in sel:
        a, b = files[i]
        xyz0 = np.load(os.path.join(args.root, a))["pcd"]
        xyz1 = np.load(os.path.join(args.root, b))["pcd"]
        pairs.append((xyz0, xyz1, np.eye(4)))  # pre-aligned => GT identity

    model = ResUNetBN2C(in_channels=1, out_channels=n_out, normalize_feature=True, conv1_kernel_size=conv1_ks).to(device).eval()

    # Warm the autotune cache once at the fixed N=cap shape (identity perm).
    id_idx7, id_idx3 = kernel_reindex(conv1_ks, np.eye(3, dtype=np.int64)), kernel_reindex(3, np.eye(3, dtype=np.int64))
    model.load_state_dict(build_wcn_sd(me_sd, conv1_ks, id_idx7, id_idx3), strict=True)
    _ = extract_capped(model, pairs[0][0], args.voxel_size, device, args.cap, seed=1)
    print(f"warmup done; {len(pairs)} pairs, cap={args.cap}, group={args.group}")

    results = []
    for name, M in octahedral_transforms(args.group):
        ridx7 = kernel_reindex(conv1_ks, M)
        ridx3 = kernel_reindex(3, M)
        sd = build_wcn_sd(me_sd, conv1_ks, ridx7, ridx3)
        model.load_state_dict(sd, strict=True)
        score = mean_inlier_ratio(model, pairs, args.voxel_size, device, args.tau, args.cap)
        results.append((score, name, M.tolist()))
        print(f"{score:.4f}  {name}", flush=True)
    results.sort(reverse=True)
    print("\n=== RANKED (mean inlier-ratio, tau=%.3f, %d pairs, cap=%d, NO rotation) ===" % (args.tau, len(pairs), args.cap))
    for score, name, M in results:
        print(f"{score:.4f}  {name}  M={M}")
    print("\nBEST:", results[0][1], results[0][2])


if __name__ == "__main__":
    main()
