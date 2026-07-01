# SPDX-License-Identifier: MIT
"""Validate the ME->WCN weight conversion on the 3DMatch VAL split (FMR proxy).

The official Princeton test benchmark URLs are dead, so this uses the val pairs
we already have as a sanity check that the converted pretrained weights produce
good geometric features. A CORRECT kernel permutation should give a high
feat-match-ratio (~0.9+); a wrong permutation collapses it to ~0. This both
confirms the port and empirically picks the right --kernel_perm.

Usage (NGC container, PYTHONPATH includes repo):
    python wcn/check_pretrained.py --wcn_ckpt <converted.pth> --root <threedmatch> --config_dir <FCGF/config>
"""

import argparse
import os
import sys
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import make_data_loader  # noqa: E402
from train import validate  # noqa: E402

from warpconvnet.models.fcgf import ResUNetBN2C


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wcn_ckpt", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--config_dir", required=True)
    p.add_argument("--voxel_size", type=float, default=0.025)
    p.add_argument("--hit_ratio_thresh", type=float, default=0.1)
    p.add_argument("--max_iter", type=int, default=200)
    args = p.parse_args()
    device = "cuda"

    ck = torch.load(args.wcn_ckpt, map_location=device, weights_only=False)
    cfg_ck = ck.get("config", {})
    conv1_ks = cfg_ck.get("conv1_kernel_size", 7)
    n_out = cfg_ck.get("model_n_out", 32)
    perm = cfg_ck.get("kernel_perm", "?")
    model = ResUNetBN2C(in_channels=1, out_channels=n_out, normalize_feature=True, conv1_kernel_size=conv1_ks)
    model.load_state_dict(ck["state_dict"], strict=True)
    model = model.to(device).eval()

    val_loader = make_data_loader(
        args.root, args.config_dir, "val", 1, voxel_size=args.voxel_size,
        num_workers=4, random_rotation=True, shuffle=False,
    )
    cfg = types.SimpleNamespace(hit_ratio_thresh=args.hit_ratio_thresh)
    fmr = validate(model, val_loader, device, cfg, max_iter=args.max_iter)
    print(f"[check] kernel_perm={perm} conv1_ks={conv1_ks} -> VAL feat_match_ratio = {fmr:.4f} "
          f"(over up to {args.max_iter} pairs)")


if __name__ == "__main__":
    main()
