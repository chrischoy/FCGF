# FCGF on WarpConvNet (`wcn/`)

MinkowskiEngine-free FCGF, built on [WarpConvNet](https://github.com/NVlabs/WarpConvNet).
The `ResUNetBN2C` descriptor network itself lives in WarpConvNet
(`warpconvnet.models.fcgf`); this directory holds the data / loss / training /
eval / conversion code around it.

| File | Purpose |
|---|---|
| `dataset.py` | 3DMatch pair dataset + collate (numpy voxelize, scipy cKDTree correspondences) |
| `loss.py` | Hardest-contrastive metric-learning loss |
| `features.py` | `extract_features(model, xyz, voxel_size)` — voxelize → forward → descriptors |
| `train.py` | Training loop (SGD + ExponentialLR, hardest-contrastive, auto-resume) |
| `eval_3dmatch.py` | Feature-Match-Recall benchmark (`--extract` / `--evaluate`) |
| `convert_me_to_wcn.py` | Convert an original MinkowskiEngine checkpoint → WarpConvNet model |
| `check_pretrained.py` | Val-split FMR proxy for a converted checkpoint |
| `smoke.py` | Synthetic forward/backward/overfit + ordering-invariant check |
| `slurm/` | SLURM launchers (env bring-up, smoke, sanity, convert-check, training) |

## Weight conversion

WarpConvNet and MinkowskiEngine enumerate the K³ sparse-conv kernel offsets in a
**transposed spatial order**, so an ME checkpoint must have its kernel weights
permuted (reverse the spatial raveling) before loading into the WarpConvNet model.
`convert_me_to_wcn.py` does this plus the name remap
(`conv1.kernel` → `conv1.0.weight`, `norm1.bn.*` → `conv1.1.*`, …). k=1 convs
(`conv1_tr`, `final`) are stored 2-D in ME and need no permutation.

```bash
python convert_me_to_wcn.py --me_ckpt 2019-08-19_06-17-41.pth --out wcn-3dmatch-32feat.pth
```

`load_fcgf_from_me(me_ckpt, kernel_perm="reverse", device)` does the same in-memory
and returns a ready model (used by `../demo.py`).

## Config (3DMatch, matches the published 0.9578-FMR model)

`ResUNetBN2C`, `in=1` (occupancy), `out=32`, `normalize_feature=True`,
`conv1_kernel_size=7`, `voxel_size=0.025`; SGD lr 0.1 / momentum 0.8 / wd 1e-4,
ExponentialLR γ=0.99, 100 epochs, batch 4; hardest-contrastive
(pos_thresh 0.1, neg_thresh 1.4, 1024 pos / 256 hn per batch-sample).
