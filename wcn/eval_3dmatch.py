# SPDX-License-Identifier: MIT
"""3DMatch Feature-Match-Recall (FMR) benchmark for the WarpConvNet FCGF port.

Port of FCGF/scripts/benchmark_3dmatch.py (feature-match-recall path), MinkowskiEngine-free.
Two stages:
  --extract : run the model on every test fragment (.ply), save {points, xyz, feature}.npz
  --evaluate: per GT pair, NN-match features, hit-ratio under GT pose; FMR = frac(hit >= tau2).

Published FCGF: FMR ~0.95 @ tau1=0.10m, tau2=0.05, 5000 keypoints.

Usage (inside NGC container, PYTHONPATH includes the WarpConvNet repo):
    python wcn/eval_3dmatch.py --extract  --model <ckpt> --source <test_dir> --target <feat_dir> --voxel_size 0.025
    python wcn/eval_3dmatch.py --evaluate --source <test_dir> --target <feat_dir> --voxel_size 0.025 --num_keypoints 5000
"""

import argparse
import glob
import logging
import os
import sys

import numpy as np
import torch
from plyfile import PlyData
from scipy.spatial import cKDTree

from warpconvnet.models.fcgf import ResUNetBN2C

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import extract_features  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")


def read_ply_xyz(path):
    ply = PlyData.read(path)
    v = ply["vertex"].data
    return np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)


def read_trajectory(filename, dim=4):
    traj = []
    with open(filename, "r") as f:
        metastr = f.readline()
        while metastr:
            metadata = list(map(int, metastr.split()))
            mat = np.zeros((dim, dim))
            for i in range(dim):
                mat[i, :] = np.fromstring(f.readline(), dtype=float, sep=" \t")
            traj.append((metadata, mat))
            metastr = f.readline()
    return traj


def fnv_hash_vec(arr):
    """FNV-1a hash of integer rows (matches ME.utils.fnv_hash_vec semantics)."""
    arr = arr.astype(np.uint64)
    hashed = np.ones(len(arr), dtype=np.uint64) * np.uint64(14695981039346656037)
    for j in range(arr.shape[1]):
        hashed *= np.uint64(1099511628211)
        hashed ^= arr[:, j]
    return hashed


def extract_all(model, source, target, voxel_size, device):
    os.makedirs(target, exist_ok=True)
    folders = sorted([d for d in glob.glob(os.path.join(source, "*")) if os.path.isdir(d) and "evaluation" not in d])
    assert folders, f"no scene folders under {source}"
    list_file = open(os.path.join(target, "list.txt"), "w")
    for fo in folders:
        base = os.path.basename(fo)
        files = sorted(glob.glob(os.path.join(fo, "*.ply")), key=lambda p: int(os.path.basename(p).split("_")[-1].split(".")[0]))
        list_file.write(f"{base} {len(files)}\n")
        logging.info(f"{base}: {len(files)} fragments")
        for i, fi in enumerate(files):
            xyz = read_ply_xyz(fi)
            xyz_down, feat = extract_features(model, xyz, voxel_size, device)
            np.savez_compressed(os.path.join(target, f"{base}_{i:03d}"), points=xyz, xyz=xyz_down, feature=feat)
    list_file.close()


def valid_feat_ratio(pts0, pts1, feat0, feat1, trans, thresh):
    pts0_t = pts0 @ trans[:3, :3].T + trans[:3, 3]
    nn = cKDTree(feat1).query(feat0, k=1)[1]
    dist = np.linalg.norm(pts0_t - pts1[nn], axis=1)
    return float(np.mean(dist < thresh))


def hit_ratio(pts0, pts1, feat0, feat1, trans_gth, tau1):
    if len(pts0) < len(pts1):
        return valid_feat_ratio(pts0, pts1, feat0, feat1, trans_gth, tau1)
    return valid_feat_ratio(pts1, pts0, feat1, feat0, np.linalg.inv(trans_gth), tau1)


def subsample_keypoints(points, xyz, feat, voxel_size, n_keypoints):
    if n_keypoints <= 0 or len(points) <= n_keypoints:
        return xyz, feat
    inds = np.random.choice(len(points), min(len(points), n_keypoints), replace=False)
    key_pts = fnv_hash_vec(np.floor(points[inds] / voxel_size))
    key_xyz = fnv_hash_vec(np.floor(xyz / voxel_size))
    keep = np.where(np.isin(key_xyz, key_pts))[0]
    return xyz[keep], feat[keep]


def evaluate(source, target, voxel_size, n_keypoints, tau1=0.1, tau2=0.05):
    sets = [l.split() for l in open(os.path.join(target, "list.txt")).read().splitlines()]
    logging.info(f"tau1={tau1} tau2={tau2} keypoints={n_keypoints}")
    recalls = []
    for set_name, _ in sets:
        gt = os.path.join(source, set_name + "-evaluation", "gt.log")
        traj = read_trajectory(gt)
        hits = []
        for metadata, pose in traj:
            i, j = metadata[0], metadata[1]
            di = np.load(os.path.join(target, f"{set_name}_{i:03d}.npz"))
            dj = np.load(os.path.join(target, f"{set_name}_{j:03d}.npz"))
            xi, fi = subsample_keypoints(di["points"], di["xyz"], di["feature"], voxel_size, n_keypoints)
            xj, fj = subsample_keypoints(dj["points"], dj["xyz"], dj["feature"], voxel_size, n_keypoints)
            hr = hit_ratio(xi, xj, fi, fj, np.linalg.inv(pose), tau1)
            hits.append(hr >= tau2)
        mean_recall = float(np.mean(hits)) if hits else 0.0
        recalls.append((set_name, mean_recall))
        logging.info(f"{set_name}: FMR={mean_recall:.4f} ({len(hits)} pairs)")
    scene_r = np.array([r for _, r in recalls])
    logging.info(f"AVERAGE FMR: {scene_r.mean():.4f} +- {scene_r.std():.4f}")
    return scene_r.mean()


def load_model(ckpt_path, conv1_kernel_size, device):
    model = ResUNetBN2C(in_channels=1, out_channels=32, normalize_feature=True, conv1_kernel_size=conv1_kernel_size)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["state_dict"])
    return model.to(device).eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--extract", action="store_true")
    p.add_argument("--evaluate", action="store_true")
    p.add_argument("--model", default=None)
    p.add_argument("--source", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--voxel_size", type=float, default=0.025)
    p.add_argument("--conv1_kernel_size", type=int, default=5)
    p.add_argument("--num_keypoints", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    np.random.seed(args.seed)
    device = "cuda"

    if args.extract:
        assert args.model, "--extract needs --model"
        model = load_model(args.model, args.conv1_kernel_size, device)
        extract_all(model, args.source, args.target, args.voxel_size, device)
    if args.evaluate:
        evaluate(args.source, args.target, args.voxel_size, args.num_keypoints)


if __name__ == "__main__":
    main()
