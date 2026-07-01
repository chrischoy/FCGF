# SPDX-License-Identifier: MIT
"""Convert an original FCGF (MinkowskiEngine) ResUNetBN2C checkpoint to the
WarpConvNet-native model :class:`warpconvnet.models.fcgf.ResUNetBN2C`.

Two differences to bridge:
  1. NAMES: ME uses `conv1.kernel`, `norm1.bn.*`, `block1.conv1.kernel`, ... ;
     our model wraps convs+BN in ConvBlock(Sequential[SparseConv3d, BN, act]) and
     ConvTrBlock(conv_tr + norm_act), so the same weights live at `conv1.0.weight`,
     `conv1.1.*`, `block1.conv1.0.weight`, `conv4_tr.conv_tr.weight`, etc.
  2. KERNEL SPATIAL ORDER: MinkowskiEngine and WarpConvNet enumerate the K^3
     kernel offsets in transposed order. PROVEN from ME source
     (MinkowskiEngine/src/kernel_region.hpp `coordinate_at`): ME decodes the flat
     kernel_index LSB-first over axes, so ME index = ax + ay*k + az*k^2 (x-fastest,
     centered, cross-correlation). WCN uses meshgrid("ij") => index = ax*k^2 +
     ay*k + az (z-fastest, centered, correlation; verified by verify_conv.py to
     6e-4). They differ only by reversing axis digit-significance (swap x<->z), i.e.
     reshape [K^3,in,out]->[k,k,k,in,out], permute spatial axes (0,1,2)->(2,1,0).
     k=1 convs need no permutation; ME stores them 2D [in,out] (bias [1,C]) ->
     reshape to [1,in,out] / [C].

The correct permutation reproduces FCGF's published feature quality; a wrong one
collapses it. `reverse` is correct for ME<->WCN (use `identity` only to A/B test).

CLI:
    python convert_me_to_wcn.py --me_ckpt <me.pth> --out <wcn.pth> [--kernel_perm reverse|identity]
Programmatic:
    from convert_me_to_wcn import load_fcgf_from_me
    model, cfg = load_fcgf_from_me("me.pth", kernel_perm="reverse", device="cuda")
"""

import argparse

import torch


def permute_kernel(w, kernel_perm="reverse"):
    """w: [K^3, in, out]. Reorder the K^3 (kernel-offset) axis; return same shape."""
    K = w.shape[0]
    k = round(K ** (1.0 / 3.0))
    assert k ** 3 == K, f"non-cubic kernel volume {K}"
    if kernel_perm == "identity" or k == 1:
        return w
    if kernel_perm == "reverse":
        w5 = w.reshape(k, k, k, w.shape[1], w.shape[2])
        w5 = w5.permute(2, 1, 0, 3, 4).contiguous()  # transpose spatial (x,y,z)->(z,y,x)
        return w5.reshape(K, w.shape[1], w.shape[2])
    raise ValueError(kernel_perm)


def me_key_to_wcn(me_key):
    """Map an ME parameter key to the WCN model's key (conv weights / BN only)."""
    if ".bn." in me_key:  # BatchNorm: <prefix>.bn.<p>
        prefix, p = me_key.split(".bn.")
        if prefix.startswith("block"):                 # blockX.norm{1,2} -> blockX.conv{1,2}.1
            blk, nm = prefix.split(".")
            return f"{blk}.conv{nm[-1]}.1.{p}"
        if prefix.endswith("_tr"):                     # normN_tr -> convN_tr.norm_act.0
            return f"{prefix.replace('norm', 'conv')}.norm_act.0.{p}"
        return f"{prefix.replace('norm', 'conv')}.1.{p}"   # normN -> convN.1
    if me_key.endswith(".kernel"):                     # conv: <prefix>.kernel
        prefix = me_key[: -len(".kernel")]
        if prefix in ("conv1_tr", "final"):
            return f"{prefix}.weight"                   # plain SparseConv3d (k=1)
        if prefix.endswith("_tr"):
            return f"{prefix}.conv_tr.weight"           # convN_tr -> convN_tr.conv_tr.weight
        return f"{prefix}.0.weight"                     # convN / blockX.convI -> .0.weight
    if me_key == "final.bias":
        return "final.bias"
    return None


def convert_me_state_dict(me_sd, kernel_perm="reverse"):
    """Return a WCN-keyed state_dict from an ME FCGF state_dict."""
    new_sd, unmapped = {}, []
    for mk, mv in me_sd.items():
        wk = me_key_to_wcn(mk)
        if wk is None:
            unmapped.append(mk)
            continue
        v = mv
        if mk.endswith(".kernel"):
            v = v.reshape(1, v.shape[0], v.shape[1]) if v.dim() == 2 else permute_kernel(v, kernel_perm)
        elif mk == "final.bias":
            v = v.reshape(-1)
        new_sd[wk] = v.clone()
    return new_sd, unmapped


def _me_cfg(ck):
    cfg = ck.get("config")
    return {
        "conv1_kernel_size": getattr(cfg, "conv1_kernel_size", 7),
        "model_n_out": getattr(cfg, "model_n_out", 32),
        "normalize_feature": getattr(cfg, "normalize_feature", True),
    }


def load_fcgf_from_me(me_ckpt, kernel_perm="reverse", device="cpu"):
    """Build a WCN ResUNetBN2C and load converted weights from an ME checkpoint.

    Returns (model, cfg_dict). Requires `easydict` to unpickle the ME config.
    """
    from warpconvnet.models.fcgf import ResUNetBN2C

    ck = torch.load(me_ckpt, map_location="cpu", weights_only=False)
    cfg = _me_cfg(ck)
    model = ResUNetBN2C(in_channels=1, out_channels=cfg["model_n_out"],
                        normalize_feature=cfg["normalize_feature"],
                        conv1_kernel_size=cfg["conv1_kernel_size"])
    new_sd, unmapped = convert_me_state_dict(ck["state_dict"], kernel_perm)
    if unmapped:
        print(f"[convert] {len(unmapped)} ME keys unmapped: {unmapped[:6]}")
    model.load_state_dict(new_sd, strict=True)  # raises loudly on any mismatch
    cfg["kernel_perm"] = kernel_perm
    return model.to(device), cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--me_ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--kernel_perm", default="reverse", choices=["reverse", "identity"])
    args = p.parse_args()
    model, cfg = load_fcgf_from_me(args.me_ckpt, args.kernel_perm)
    print(f"converted (conv1_ks={cfg['conv1_kernel_size']} n_out={cfg['model_n_out']} "
          f"perm={cfg['kernel_perm']}); strict load OK")
    torch.save({"state_dict": model.state_dict(), "config": cfg}, args.out)
    print(f"saved WCN checkpoint -> {args.out}")


if __name__ == "__main__":
    main()
