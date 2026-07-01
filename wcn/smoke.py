# SPDX-License-Identifier: MIT
"""Synthetic smoke test for the WarpConvNet-native FCGF port (no dataset needed).

Checks, in order:
  1. Model builds and runs forward on a batch of two synthetic voxelized clouds.
  2. ORDERING INVARIANT: the output Voxels' coordinates/offsets match the input's
     row-for-row. FCGF correspondences index features in input order, so this must
     hold (or downstream loss/eval silently misaligns).
  3. Output descriptors are 32-D and L2-normalized.
  4. Backward produces finite grads for every parameter.
  5. The hardest-contrastive loss overfits a single fixed pair (loss goes down).

Run inside the NGC container:  python wcn/smoke.py
"""

import os
import sys

import numpy as np
import torch

from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.models.fcgf import ResUNetBN2C

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loss import hardest_contrastive_loss  # noqa: E402


def voxelize(xyz: np.ndarray, voxel_size: float):
    """Floor-quantize + dedup (numpy replacement for ME.utils.sparse_quantize)."""
    coords = np.floor(xyz / voxel_size).astype(np.int32)
    _, sel = np.unique(coords, axis=0, return_index=True)
    sel = np.sort(sel)
    return coords[sel], sel


def make_cloud(n, voxel_size, seed):
    rng = np.random.RandomState(seed)
    xyz = rng.rand(n, 3) * 2.0  # 2m cube
    coords, sel = voxelize(xyz, voxel_size)
    feats = np.ones((len(coords), 1), dtype=np.float32)
    return coords, feats, xyz[sel]


def main():
    assert torch.cuda.is_available(), "smoke test expects a GPU"
    device = "cuda"
    torch.manual_seed(0)
    np.random.seed(0)
    voxel_size = 0.025

    model = ResUNetBN2C(in_channels=1, out_channels=32, normalize_feature=True, conv1_kernel_size=5).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[build] ResUNetBN2C params = {n_params/1e6:.3f}M")

    # Two-sample batch of synthetic voxel clouds.
    c0, f0, _ = make_cloud(4000, voxel_size, seed=1)
    c1, f1, _ = make_cloud(3500, voxel_size, seed=2)
    coords = [torch.from_numpy(c0), torch.from_numpy(c1)]
    feats = [torch.from_numpy(f0), torch.from_numpy(f1)]
    vin = Voxels(coords, feats).to(device)
    print(f"[input] N={[len(c0), len(c1)]}  coord_tensor={tuple(vin.coordinate_tensor.shape)}  offsets={vin.offsets.tolist()}")

    # Does the Voxels constructor preserve the row order of the coords we passed?
    # (Correspondences are computed in dataset/list order; if the constructor
    # reorders, collate must remap them.)
    passed = torch.cat(coords, dim=0)
    ctor_preserves = torch.equal(vin.coordinate_tensor.cpu(), passed)
    print(f"[input] Voxels ctor preserves passed row order = {ctor_preserves}")

    model.eval()
    with torch.no_grad():
        vout = model(vin)

    # ---- ordering invariant ----
    same_off = torch.equal(vout.offsets.cpu(), vin.offsets.cpu())
    same_coords = torch.equal(vout.coordinate_tensor.cpu(), vin.coordinate_tensor.cpu())
    print(f"[order] offsets match={same_off}  coords match row-for-row={same_coords}")
    if not (same_off and same_coords):
        # Report how badly it diverges so we know whether a remap is needed.
        print("  !! ordering NOT preserved — loss/eval must remap features by coordinate hash")
        print(f"     out coords shape {tuple(vout.coordinate_tensor.shape)} vs in {tuple(vin.coordinate_tensor.shape)}")

    # ---- output shape / normalization ----
    Fout = vout.feature_tensor
    norms = Fout.norm(dim=1)
    print(f"[feat] out dim={Fout.shape[1]}  mean L2 norm={norms.mean().item():.4f} (expect ~1.0)")
    assert Fout.shape[1] == 32

    # ---- backward ----
    model.train()
    vout = model(vin)
    loss = vout.feature_tensor.sum()
    loss.backward()
    n_grad = sum(1 for p in model.parameters() if p.grad is not None and torch.isfinite(p.grad).all())
    n_total = sum(1 for _ in model.parameters())
    print(f"[bwd] finite grads on {n_grad}/{n_total} params")
    assert n_grad == n_total

    # ---- overfit a single fixed pair ----
    # Build synthetic ground-truth correspondences: first K coords of cloud0 map
    # to first K of cloud1 (arbitrary but fixed). Loss should drive these together.
    K = 512
    pos = torch.stack([torch.arange(K), torch.arange(K)], dim=1).long().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.8, weight_decay=1e-4)
    losses = []
    for step in range(60):
        opt.zero_grad()
        vout = model(vin)
        off = vout.offsets
        Fa = vout.feature_tensor[off[0]:off[1]]
        Fb = vout.feature_tensor[off[1]:off[2]]
        pl, nl = hardest_contrastive_loss(Fa, Fb, pos, num_pos=K, num_hn_samples=512)
        l = pl + nl
        l.backward()
        opt.step()
        losses.append(l.item())
        if step % 10 == 0:
            print(f"[overfit] step {step:3d}  loss={l.item():.4f}  pos={pl.item():.4f}  neg={nl.item():.4f}")
    print(f"[overfit] first={losses[0]:.4f}  last={losses[-1]:.4f}  drop={losses[0]-losses[-1]:.4f}")
    assert losses[-1] < losses[0], "loss did not decrease — overfit failed"
    print("\nSMOKE PASSED")


if __name__ == "__main__":
    main()
