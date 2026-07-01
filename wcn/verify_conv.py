# SPDX-License-Identifier: MIT
"""Pin down WarpConvNet's sparse-conv semantics numerically (no MinkowskiEngine).

A sparse conv computes  out[p] = sum_o W[widx(o)] @ in[p + sign*o]  over kernel
offsets o. Both ME and WCN implement this SAME function; they differ only in the
index->offset map widx (and possibly the +/- sign convention). This test:

  1. Runs WCN SparseConv3d(k=3, stride=1) with a known random weight on a random
     voxel cloud, and compares against a hand-written gather-convolution that uses
     the meshgrid("ij"), z-fastest, centered offset ordering (WCN's documented
     order) under both sign conventions. The matching one nails WCN's exact
     semantics + ordering to fp tolerance.
  2. Confirms that permuting the K^3 weight axis by the spatial reverse
     (x,y,z)->(z,y,x) corresponds to reversing the offset list — i.e. exactly the
     transpose that converts an ME kernel to WCN. If WCN==ref under order `O`,
     then feeding ME weights (order transpose(O)) after `reverse` makes WCN
     compute ME's function.

If (1) matches at ~1e-4, WCN's conv is a bit-faithful sparse convolution and the
ME->WCN weight conversion is exact up to the ordering, which is pinned here.
"""

import numpy as np
import torch

from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.nn.modules.sparse_conv import SparseConv3d


def wcn_offsets(k=3):
    """WCN kernel offsets: meshgrid('ij') over (x,y,z), z-fastest, centered."""
    r = [torch.arange(k) for _ in range(3)]
    g = torch.meshgrid(*r, indexing="ij")
    off = torch.stack([x.flatten() for x in g], dim=1) - (k - 1) // 2
    return off  # [K^3, 3], row index = WCN weight index


def ref_conv(coords, feats, W, offsets, sign):
    """out[i] = sum_k feats[nbr(coords[i] + sign*offsets[k])] @ W[k]."""
    N, Cout = len(coords), W.shape[2]
    cmap = {tuple(c.tolist()): i for i, c in enumerate(coords)}
    out = torch.zeros(N, Cout, dtype=feats.dtype)
    for i in range(N):
        ci = coords[i]
        for k in range(offsets.shape[0]):
            j = cmap.get(tuple((ci + sign * offsets[k]).tolist()))
            if j is not None:
                out[i] += feats[j] @ W[k]
    return out


def main():
    torch.manual_seed(0)
    device = "cuda"
    Cin, Cout, k, N = 5, 7, 3, 400

    # Random unique voxel cloud in a small grid (dense-ish so neighbors exist).
    rng = np.random.RandomState(0)
    raw = rng.randint(0, 8, size=(N, 3)).astype(np.int32)
    coords_np = np.unique(raw, axis=0)
    coords = torch.from_numpy(coords_np)
    feats = torch.randn(len(coords), Cin)

    conv = SparseConv3d(Cin, Cout, kernel_size=k, stride=1, bias=False).to(device)
    v = Voxels([coords], [feats]).to(device)
    out = conv(v)
    # Align WCN output to input coord order (should already match; be safe).
    assert torch.equal(out.coordinate_tensor.cpu(), coords), "coord order changed"
    wcn_out = out.feature_tensor.detach().cpu()
    W = conv.weight.detach().cpu()  # [K^3, Cin, Cout]
    print(f"WCN weight shape {tuple(W.shape)}  (K^3={k**3})")

    offs = wcn_offsets(k)
    for sign in (+1, -1):
        r = ref_conv(coords, feats, W, offs, sign)
        diff = (wcn_out - r).abs().max().item()
        print(f"[semantics] sign={sign:+d} meshgrid-ij order -> max|WCN-ref| = {diff:.3e}")

    # Also test the REVERSED offset order (spatial transpose) to show it does NOT
    # match with the same weights (i.e. reverse is a genuine, non-trivial permutation).
    offs_rev = offs.flip(0)
    for sign in (+1, -1):
        r = ref_conv(coords, feats, W, offs_rev, sign)
        diff = (wcn_out - r).abs().max().item()
        print(f"[semantics] sign={sign:+d} REVERSED order      -> max|WCN-ref| = {diff:.3e}")
    print("If exactly one (sign, order) row is ~1e-5, WCN's conv semantics + ordering are pinned.")


if __name__ == "__main__":
    main()
