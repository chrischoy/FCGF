"""FCGF demo on the WarpConvNet backend.

Downloads an original (MinkowskiEngine) FCGF checkpoint from the HuggingFace
mirror, converts it on the fly to the WarpConvNet-native model
(:class:`warpconvnet.models.fcgf.ResUNetBN2C`), extracts geometric features on a
point cloud, and visualizes them (features -> colors). No MinkowskiEngine needed.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "wcn"))
from convert_me_to_wcn import load_fcgf_from_me  # noqa: E402
from features import extract_features  # noqa: E402

HF = "https://huggingface.co/chrischoy/FCGF/resolve/main"


def _download(fname, url):
    if not os.path.isfile(fname):
        from urllib.request import urlretrieve
        print(f"Downloading {fname} ...")
        urlretrieve(url, fname)


def load_fcgf_model(ckpt_path, kernel_perm, device):
    """Load a checkpoint as the WarpConvNet FCGF model. Accepts either an original
    MinkowskiEngine checkpoint (converted on the fly) or one already converted by
    wcn/convert_me_to_wcn.py."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", {})
    if any(k.endswith(".kernel") for k in sd):  # original ME checkpoint
        return load_fcgf_from_me(ckpt_path, kernel_perm=kernel_perm, device=device)
    from warpconvnet.models.fcgf import ResUNetBN2C  # already-converted WCN checkpoint
    cfg = ck.get("config", {})
    model = ResUNetBN2C(in_channels=1, out_channels=cfg.get("model_n_out", 32),
                        normalize_feature=True, conv1_kernel_size=cfg.get("conv1_kernel_size", 7))
    model.load_state_dict(sd, strict=True)
    return model.to(device), cfg


def demo(config):
    import open3d as o3d
    from util.visualization import get_colored_point_cloud_feature

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_fcgf_model(config.model, config.kernel_perm, device)
    model.eval()
    print(f"loaded FCGF: conv1_kernel_size={cfg['conv1_kernel_size']} n_out={cfg['model_n_out']}")

    pcd = o3d.io.read_point_cloud(config.input)
    xyz_down, feature = extract_features(
        model, xyz=np.asarray(pcd.points), voxel_size=config.voxel_size, device=device
    )

    vis_pcd = o3d.geometry.PointCloud()
    vis_pcd.points = o3d.utility.Vector3dVector(xyz_down)
    vis_pcd = get_colored_point_cloud_feature(vis_pcd, feature, config.voxel_size)
    o3d.visualization.draw_geometries([vis_pcd])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default="redkitchen-20.ply", help="point cloud file")
    parser.add_argument("-m", "--model", default="ResUNetBN2C-16feat-3conv.pth",
                        help="ORIGINAL (MinkowskiEngine) FCGF checkpoint; converted to WCN on load")
    parser.add_argument("--kernel_perm", default="reverse", choices=["reverse", "identity"],
                        help="ME<->WCN kernel spatial-order permutation (reverse is correct)")
    parser.add_argument("--voxel_size", default=0.025, type=float)
    config = parser.parse_args()

    _download(config.model, f"{HF}/2019-09-18_14-15-59.pth")  # ResUNetBN2C 16-feat 3conv
    _download(config.input, f"{HF}/redkitchen-20.ply")
    demo(config)
