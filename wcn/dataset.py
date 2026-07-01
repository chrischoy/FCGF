# SPDX-License-Identifier: MIT
"""3DMatch pair dataset for the WarpConvNet FCGF port (MinkowskiEngine-free).

Port of ``FCGF/lib/data_loaders.py::ThreeDMatchPairDataset``. Differences:
- ``ME.utils.sparse_quantize`` -> numpy floor-quantize + ``np.unique`` (first
  occurrence, original relative order preserved).
- open3d KDTree correspondence search -> ``scipy.spatial.cKDTree``.
- collate returns per-sample lists (built into ``Voxels`` in the training loop)
  plus correspondences offset into the concatenated feature order.

Training pairs are pre-aligned in a common world frame (GT = identity); a random
SE(3) is applied to each cloud independently and ``trans = T1 @ inv(T0)`` recovers
the relative pose used to find positive correspondences.
"""

import glob
import os
import random
from typing import List, Optional

import numpy as np
import torch
from scipy.linalg import expm, norm
from scipy.spatial import cKDTree
from torch.utils.data import Dataset


# ----------------------------------------------------------------------------
# geometry helpers
# ----------------------------------------------------------------------------
def _M(axis, theta):
    return expm(np.cross(np.eye(3), axis / norm(axis) * theta))


def sample_random_trans(pcd, randg, rotation_range=360):
    T = np.eye(4)
    R = _M(randg.rand(3) - 0.5, rotation_range * np.pi / 180.0 * (randg.rand(1) - 0.5))
    T[:3, :3] = R
    T[:3, 3] = R.dot(-np.mean(pcd, axis=0))
    return T


def apply_transform(pts, trans):
    R = trans[:3, :3]
    T = trans[:3, 3]
    return pts @ R.T + T


def voxel_quantize(xyz: np.ndarray, voxel_size: float):
    """Return (int coords, sel) where sel indexes the first point in each voxel.

    Original relative order is preserved (``np.unique`` sorts, so we re-sort sel).
    """
    coords = np.floor(xyz / voxel_size).astype(np.int32)
    _, sel = np.unique(coords, axis=0, return_index=True)
    sel = np.sort(sel)
    return coords[sel], sel


def get_matching_indices(xyz0_trans: np.ndarray, xyz1: np.ndarray, search_radius: float) -> np.ndarray:
    """All (i, j) with ``||xyz0_trans[i] - xyz1[j]|| <= search_radius`` (many-to-many).

    ``xyz0_trans`` must already be transformed into ``xyz1``'s frame. Matches
    FCGF's open3d radius search semantics.
    """
    tree = cKDTree(xyz1)
    nbrs = tree.query_ball_point(xyz0_trans, r=search_radius)
    matches = []
    for i, js in enumerate(nbrs):
        for j in js:
            matches.append((i, j))
    if len(matches) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    return np.asarray(matches, dtype=np.int64)


# ----------------------------------------------------------------------------
# dataset
# ----------------------------------------------------------------------------
class ThreeDMatchPairDataset(Dataset):
    OVERLAP_RATIO = 0.3
    DATA_FILES = {
        "train": "config/train_3dmatch.txt",
        "val": "config/val_3dmatch.txt",
        "test": "config/test_3dmatch.txt",
    }

    def __init__(
        self,
        phase: str,
        root: str,
        config_dir: str,
        voxel_size: float = 0.025,
        positive_pair_search_voxel_size_multiplier: float = 1.5,
        random_rotation: bool = True,
        rotation_range: float = 360,
        random_scale: bool = False,
        min_scale: float = 0.8,
        max_scale: float = 1.2,
        manual_seed: bool = False,
    ):
        self.phase = phase
        self.root = root
        self.voxel_size = voxel_size
        self.matching_search_voxel_size = voxel_size * positive_pair_search_voxel_size_multiplier
        self.random_rotation = random_rotation
        self.rotation_range = rotation_range
        self.random_scale = random_scale
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.randg = np.random.RandomState()
        if manual_seed:
            self.reset_seed()

        self.files: List[List[str]] = []
        list_file = os.path.join(config_dir, os.path.basename(self.DATA_FILES[phase]))
        subset_names = open(list_file).read().split()
        for name in subset_names:
            fname = name + "*%.2f.txt" % self.OVERLAP_RATIO
            fnames_txt = glob.glob(os.path.join(root, fname))
            assert len(fnames_txt) > 0, f"Make sure {root} has data {fname}"
            for fname_txt in fnames_txt:
                with open(fname_txt) as f:
                    content = f.readlines()
                for line in content:
                    pair = line.strip().split()
                    self.files.append([pair[0], pair[1]])

    def reset_seed(self, seed=0):
        self.randg.seed(seed)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file0 = os.path.join(self.root, self.files[idx][0])
        file1 = os.path.join(self.root, self.files[idx][1])
        data0 = np.load(file0)
        data1 = np.load(file1)
        xyz0 = data0["pcd"]
        xyz1 = data1["pcd"]

        matching_search_voxel_size = self.matching_search_voxel_size

        if self.random_scale and random.random() < 0.95:
            scale = self.min_scale + (self.max_scale - self.min_scale) * random.random()
            matching_search_voxel_size *= scale
            xyz0 = scale * xyz0
            xyz1 = scale * xyz1

        if self.random_rotation:
            T0 = sample_random_trans(xyz0, self.randg, self.rotation_range)
            T1 = sample_random_trans(xyz1, self.randg, self.rotation_range)
            trans = T1 @ np.linalg.inv(T0)
            xyz0 = apply_transform(xyz0, T0)
            xyz1 = apply_transform(xyz1, T1)
        else:
            trans = np.identity(4)

        # Voxelize (quantize + dedup).
        coords0, sel0 = voxel_quantize(xyz0, self.voxel_size)
        coords1, sel1 = voxel_quantize(xyz1, self.voxel_size)
        pts0 = xyz0[sel0]
        pts1 = xyz1[sel1]

        # Positive correspondences on the voxelized points (cloud0 -> cloud1 frame).
        pts0_trans = apply_transform(pts0, trans)
        matches = get_matching_indices(pts0_trans, pts1, matching_search_voxel_size)

        feats0 = np.ones((len(coords0), 1), dtype=np.float32)
        feats1 = np.ones((len(coords1), 1), dtype=np.float32)

        return {
            "coords0": torch.from_numpy(coords0),
            "coords1": torch.from_numpy(coords1),
            "feats0": torch.from_numpy(feats0),
            "feats1": torch.from_numpy(feats1),
            "xyz0": torch.from_numpy(pts0).float(),
            "xyz1": torch.from_numpy(pts1).float(),
            "matches": torch.from_numpy(matches),
            "trans": torch.from_numpy(trans).float(),
        }


def collate_pair_fn(batch):
    """Collate to per-sample lists + correspondences offset into concatenated order."""
    coords0 = [b["coords0"] for b in batch]
    coords1 = [b["coords1"] for b in batch]
    feats0 = [b["feats0"] for b in batch]
    feats1 = [b["feats1"] for b in batch]
    xyz0 = [b["xyz0"] for b in batch]
    xyz1 = [b["xyz1"] for b in batch]
    trans = torch.stack([b["trans"] for b in batch], dim=0)

    matches_batch = []
    off0, off1 = 0, 0
    len_batch = []
    for b in batch:
        m = b["matches"].clone()
        if len(m) > 0:
            m[:, 0] += off0
            m[:, 1] += off1
        matches_batch.append(m)
        n0, n1 = len(b["coords0"]), len(b["coords1"])
        len_batch.append((n0, n1))
        off0 += n0
        off1 += n1
    matches_batch = torch.cat(matches_batch, dim=0) if matches_batch else torch.zeros((0, 2), dtype=torch.long)

    return {
        "coords0": coords0,
        "coords1": coords1,
        "feats0": feats0,
        "feats1": feats1,
        "xyz0": xyz0,
        "xyz1": xyz1,
        "correspondences": matches_batch,
        "trans": trans,
        "len_batch": len_batch,
    }


def make_data_loader(
    root: str,
    config_dir: str,
    phase: str,
    batch_size: int,
    voxel_size: float = 0.025,
    num_workers: int = 4,
    shuffle: Optional[bool] = None,
    random_rotation: bool = True,
    random_scale: bool = False,
    rotation_range: float = 360,
    repeat: bool = False,
):
    if shuffle is None:
        shuffle = phase != "test"
    dset = ThreeDMatchPairDataset(
        phase=phase,
        root=root,
        config_dir=config_dir,
        voxel_size=voxel_size,
        random_rotation=random_rotation,
        random_scale=random_scale,
        rotation_range=rotation_range,
    )
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_pair_fn,
        pin_memory=False,
        drop_last=True,
    )
    return loader
