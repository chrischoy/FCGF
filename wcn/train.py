# SPDX-License-Identifier: MIT
"""Train the WarpConvNet-native FCGF (ResUNetBN2C) on 3DMatch.

Faithful to FCGF's HardestContrastiveLossTrainer training config:
SGD(lr=0.1, momentum=0.8, wd=1e-4), ExponentialLR(gamma=0.99), 100 epochs,
batch_size=4, voxel_size=0.025. Checkpoints model_last.pth every epoch (for
4h-wall requeue) and model_best.pth on val feat-match-ratio.

Usage (inside NGC container, PYTHONPATH includes the WarpConvNet repo):
    python wcn/train.py --root <threedmatch_dir> --out_dir <exp_dir> [flags]
"""

import argparse
import logging
import os
import sys
import time

import numpy as np
import torch

from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.models.fcgf import ResUNetBN2C

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from dataset import make_data_loader  # noqa: E402
from loss import hardest_contrastive_loss  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")


# ----------------------------------------------------------------------------
# feature alignment
# ----------------------------------------------------------------------------
def cat_order_bcoords(coords_list, device):
    """Batch-indexed coords ``[N,4]`` in the concatenated (dataset) order that
    ``correspondences`` index into."""
    parts = []
    for b, c in enumerate(coords_list):
        bidx = torch.full((len(c), 1), b, dtype=c.dtype)
        parts.append(torch.cat([bidx, c], dim=1))
    return torch.cat(parts, dim=0).to(device)


def features_in_cat_order(vout: Voxels, desired_bcoords: torch.Tensor) -> torch.Tensor:
    """Reorder ``vout`` features so row i corresponds to ``desired_bcoords[i]``.

    No-op fast path when the model already preserves the input order; otherwise
    a robust permutation via ``torch.unique`` on the batch-indexed coordinates.
    """
    src_bcoords = vout.batch_indexed_coordinates
    feats = vout.feature_tensor
    desired_bcoords = desired_bcoords.to(src_bcoords.dtype)
    if src_bcoords.shape == desired_bcoords.shape and torch.equal(src_bcoords, desired_bcoords):
        return feats
    combined = torch.cat([desired_bcoords, src_bcoords], dim=0)
    _, inv = torch.unique(combined, dim=0, return_inverse=True)
    N = desired_bcoords.shape[0]
    inv_d, inv_s = inv[:N], inv[N:]
    pos = torch.empty(int(inv.max()) + 1, dtype=torch.long, device=feats.device)
    pos[inv_s] = torch.arange(len(inv_s), device=feats.device)
    return feats[pos[inv_d]]


def forward_pair(model, batch, device):
    """Run both clouds; return (F0, F1) aligned to correspondence indices."""
    v0 = Voxels(batch["coords0"], batch["feats0"]).to(device)
    v1 = Voxels(batch["coords1"], batch["feats1"]).to(device)
    out0 = model(v0)
    out1 = model(v1)
    d0 = cat_order_bcoords(batch["coords0"], device)
    d1 = cat_order_bcoords(batch["coords1"], device)
    F0 = features_in_cat_order(out0, d0)
    F1 = features_in_cat_order(out1, d1)
    return F0, F1


# ----------------------------------------------------------------------------
# train / eval
# ----------------------------------------------------------------------------
def save_ckpt(path, epoch, model, optimizer, scheduler, best_val, best_epoch):
    torch.save(
        {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val": best_val,
            "best_val_epoch": best_epoch,
        },
        path,
    )


@torch.no_grad()
def validate(model, loader, device, cfg, max_iter=400, subsample=5000):
    """Feature-match-ratio proxy (FCGF _valid_epoch): subsample keypoints, NN-match
    features 0->1, hit ratio = frac within hit_ratio_thresh under GT; FMR = frac(hit>0.05)."""
    model.eval()
    hit_thresh = cfg.hit_ratio_thresh
    fmrs = []
    for n, batch in enumerate(loader):
        if n >= max_iter:
            break
        F0, F1 = forward_pair(model, batch, device)
        xyz0 = torch.cat(batch["xyz0"], 0).to(device)
        xyz1 = torch.cat(batch["xyz1"], 0).to(device)
        # Subsample to bound the pairwise distance matrix (matches FCGF find_corr).
        if len(F0) > subsample:
            i0 = torch.randperm(len(F0), device=device)[:subsample]
            F0, xyz0 = F0[i0], xyz0[i0]
        if len(F1) > subsample:
            i1 = torch.randperm(len(F1), device=device)[:subsample]
            F1, xyz1 = F1[i1], xyz1[i1]
        trans = batch["trans"][0].to(device)
        nn = torch.cdist(F0, F1).argmin(dim=1)
        xyz0_t = xyz0 @ trans[:3, :3].T + trans[:3, 3]
        dist = (xyz0_t - xyz1[nn]).norm(dim=1)
        hit = (dist < hit_thresh).float().mean().item()
        fmrs.append(hit > 0.05)
    model.train()
    return float(np.mean(fmrs)) if fmrs else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="threedmatch data dir")
    p.add_argument("--config_dir", default=None, help="dir with train/val_3dmatch.txt (default: FCGF/config)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--voxel_size", type=float, default=0.025)
    p.add_argument("--conv1_kernel_size", type=int, default=5)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.8)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--exp_gamma", type=float, default=0.99)
    p.add_argument("--max_epoch", type=int, default=100)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--num_pos_per_batch", type=int, default=1024)
    p.add_argument("--num_hn_samples_per_batch", type=int, default=256)
    p.add_argument("--pos_thresh", type=float, default=0.1)
    p.add_argument("--neg_thresh", type=float, default=1.4)
    p.add_argument("--neg_weight", type=float, default=1.0)
    p.add_argument("--hit_ratio_thresh", type=float, default=0.1)
    p.add_argument("--stat_freq", type=int, default=40)
    p.add_argument("--val_epoch_freq", type=int, default=1)
    p.add_argument("--val_max_iter", type=int, default=400)
    p.add_argument("--max_train_iters", type=int, default=0, help="cap iters/epoch (0=all); for quick sanity runs")
    p.add_argument("--resume", default=None)
    cfg = p.parse_args()

    if cfg.config_dir is None:
        cfg.config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")
    os.makedirs(cfg.out_dir, exist_ok=True)
    device = "cuda"

    model = ResUNetBN2C(
        in_channels=1, out_channels=32, normalize_feature=True, conv1_kernel_size=cfg.conv1_kernel_size
    ).to(device)
    logging.info(f"model params = {sum(p.numel() for p in model.parameters())/1e6:.3f}M")

    optimizer = torch.optim.SGD(
        model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, cfg.exp_gamma)

    start_epoch = 1
    best_val, best_epoch = -1.0, -1
    # Auto-resume from model_last if present (4h-wall requeue) or explicit --resume.
    resume_path = cfg.resume or os.path.join(cfg.out_dir, "model_last.pth")
    if os.path.isfile(resume_path):
        state = torch.load(resume_path, map_location=device)
        model.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = state["epoch"] + 1
        best_val = state.get("best_val", -1.0)
        best_epoch = state.get("best_val_epoch", -1)
        logging.info(f"resumed from {resume_path} @ epoch {state['epoch']} (best_val={best_val:.4f})")

    train_loader = make_data_loader(
        cfg.root, cfg.config_dir, "train", cfg.batch_size, voxel_size=cfg.voxel_size,
        num_workers=cfg.num_workers, random_rotation=True,
    )
    val_loader = make_data_loader(
        cfg.root, cfg.config_dir, "val", 1, voxel_size=cfg.voxel_size,
        num_workers=1, random_rotation=True, shuffle=False,
    )
    logging.info(f"train pairs={len(train_loader.dataset)}  val pairs={len(val_loader.dataset)}")

    num_pos = cfg.num_pos_per_batch * cfg.batch_size
    num_hn = cfg.num_hn_samples_per_batch * cfg.batch_size

    for epoch in range(start_epoch, cfg.max_epoch + 1):
        model.train()
        t0 = time.time()
        for it, batch in enumerate(train_loader):
            if cfg.max_train_iters and it >= cfg.max_train_iters:
                break
            if len(batch["correspondences"]) < 1:
                continue
            optimizer.zero_grad()
            F0, F1 = forward_pair(model, batch, device)
            pos = batch["correspondences"].to(device)
            pos_loss, neg_loss = hardest_contrastive_loss(
                F0, F1, pos, num_pos=num_pos, num_hn_samples=num_hn,
                pos_thresh=cfg.pos_thresh, neg_thresh=cfg.neg_thresh,
            )
            loss = pos_loss + cfg.neg_weight * neg_loss
            loss.backward()
            optimizer.step()
            if it % cfg.stat_freq == 0:
                logging.info(
                    f"ep {epoch} [{it}/{len(train_loader)}] loss={loss.item():.4f} "
                    f"pos={pos_loss.item():.4f} neg={neg_loss.item():.4f} lr={scheduler.get_last_lr()[0]:.4e}"
                )
        scheduler.step()
        save_ckpt(os.path.join(cfg.out_dir, "model_last.pth"), epoch, model, optimizer, scheduler, best_val, best_epoch)
        logging.info(f"epoch {epoch} done in {time.time()-t0:.1f}s")

        if epoch % cfg.val_epoch_freq == 0:
            fmr = validate(model, val_loader, device, cfg, max_iter=cfg.val_max_iter)
            logging.info(f"[val] epoch {epoch} feat_match_ratio={fmr:.4f} (best={best_val:.4f}@{best_epoch})")
            if fmr > best_val:
                best_val, best_epoch = fmr, epoch
                save_ckpt(os.path.join(cfg.out_dir, "model_best.pth"), epoch, model, optimizer, scheduler, best_val, best_epoch)
                logging.info(f"[val] new best {best_val:.4f}")


if __name__ == "__main__":
    main()
