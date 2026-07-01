# SPDX-License-Identifier: MIT
"""Shared feature extraction for the WarpConvNet-native FCGF model.

Voxelize a raw point cloud, run :class:`warpconvnet.models.fcgf.ResUNetBN2C`,
and return the downsampled points + their descriptors — the WarpConvNet
replacement for FCGF/util/misc.py::extract_features (which used MinkowskiEngine).
"""

import numpy as np
import torch

from warpconvnet.geometry.types.voxels import Voxels


@torch.no_grad()
def extract_features(model, xyz, voxel_size=0.025, device="cuda"):
    """Return (xyz_down [M,3], features [M,C]) for a raw point cloud ``xyz`` [N,3].

    Voxelizes with a floor-quantize + first-in-voxel dedup (the MinkowskiEngine-free
    equivalent of ``ME.utils.sparse_quantize``). Feature is 1-D occupancy (all ones).
    """
    xyz = np.asarray(xyz)
    coords = np.floor(xyz / voxel_size).astype(np.int32)
    _, sel = np.unique(coords, axis=0, return_index=True)
    sel = np.sort(sel)
    c = torch.from_numpy(coords[sel])
    f = torch.ones((len(sel), 1), dtype=torch.float32)
    vox = Voxels([c], [f]).to(device)
    out = model(vox)
    return xyz[sel], out.feature_tensor.detach().cpu().numpy()
