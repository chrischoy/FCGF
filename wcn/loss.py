# SPDX-License-Identifier: MIT
"""Hardest-contrastive metric-learning loss for FCGF (WarpConvNet port).

Faithful reimplementation of
``FCGF/lib/trainer.py::HardestContrastiveLossTrainer.contrastive_hardest_negative_loss``.
Pure PyTorch/NumPy — no MinkowskiEngine dependency. Features are expected in
input-coordinate order so that ``positive_pairs`` (built in the data loader on
the input coordinates) index the right rows.
"""

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


def pdist(A: torch.Tensor, B: torch.Tensor, dist_type: str = "L2") -> torch.Tensor:
    """Pairwise distance matrix (``[len(A), len(B)]``). Matches FCGF/lib/metrics.py."""
    if dist_type == "L2":
        D2 = torch.sum((A.unsqueeze(1) - B.unsqueeze(0)).pow(2), 2)
        return torch.sqrt(D2 + 1e-7)
    elif dist_type == "SquareL2":
        return torch.sum((A.unsqueeze(1) - B.unsqueeze(0)).pow(2), 2)
    raise NotImplementedError(dist_type)


def _hash(arr, M: int) -> np.ndarray:
    """Row hash used to test whether a sampled negative is secretly a positive."""
    if isinstance(arr, np.ndarray):
        N, D = arr.shape
    else:
        N, D = len(arr[0]), len(arr)
    hash_vec = np.zeros(N, dtype=np.int64)
    for d in range(D):
        if isinstance(arr, np.ndarray):
            hash_vec += arr[:, d] * M ** d
        else:
            hash_vec += arr[d] * M ** d
    return hash_vec


def hardest_contrastive_loss(
    F0: torch.Tensor,
    F1: torch.Tensor,
    positive_pairs: torch.Tensor,
    num_pos: int = 4096,
    num_hn_samples: int = 1024,
    pos_thresh: float = 0.1,
    neg_thresh: float = 1.4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(pos_loss, neg_loss)``.

    Parameters
    ----------
    F0, F1 : ``[N0, C]`` / ``[N1, C]`` features (input-coordinate order).
    positive_pairs : ``[P, 2]`` long tensor of (idx-in-F0, idx-in-F1).
    num_pos : max number of positive pairs to keep per step.
    num_hn_samples : size of the random candidate pool mined for hardest negatives.
    """
    N0, N1 = len(F0), len(F1)
    N_pos_pairs = len(positive_pairs)
    hash_seed = max(N0, N1)
    sel0 = np.random.choice(N0, min(N0, num_hn_samples), replace=False)
    sel1 = np.random.choice(N1, min(N1, num_hn_samples), replace=False)

    if N_pos_pairs > num_pos:
        pos_sel = np.random.choice(N_pos_pairs, num_pos, replace=False)
        sample_pos_pairs = positive_pairs[pos_sel]
    else:
        sample_pos_pairs = positive_pairs

    # Candidate negatives.
    subF0, subF1 = F0[sel0], F1[sel1]

    pos_ind0 = sample_pos_pairs[:, 0].long()
    pos_ind1 = sample_pos_pairs[:, 1].long()
    posF0, posF1 = F0[pos_ind0], F1[pos_ind1]

    D01 = pdist(posF0, subF1, dist_type="L2")
    D10 = pdist(posF1, subF0, dist_type="L2")

    D01min, D01ind = D01.min(1)
    D10min, D10ind = D10.min(1)

    if not isinstance(positive_pairs, np.ndarray):
        positive_pairs_np = np.asarray(positive_pairs.cpu().numpy(), dtype=np.int64)
    else:
        positive_pairs_np = positive_pairs

    pos_keys = _hash(positive_pairs_np, hash_seed)

    D01ind = sel1[D01ind.cpu().numpy()]
    D10ind = sel0[D10ind.cpu().numpy()]
    neg_keys0 = _hash([pos_ind0.cpu().numpy(), D01ind], hash_seed)
    neg_keys1 = _hash([D10ind, pos_ind1.cpu().numpy()], hash_seed)

    mask0 = torch.from_numpy(np.logical_not(np.isin(neg_keys0, pos_keys, assume_unique=False)))
    mask1 = torch.from_numpy(np.logical_not(np.isin(neg_keys1, pos_keys, assume_unique=False)))

    pos_loss = F.relu((posF0 - posF1).pow(2).sum(1) - pos_thresh)
    neg_loss0 = F.relu(neg_thresh - D01min[mask0]).pow(2)
    neg_loss1 = F.relu(neg_thresh - D10min[mask1]).pow(2)
    return pos_loss.mean(), (neg_loss0.mean() + neg_loss1.mean()) / 2
