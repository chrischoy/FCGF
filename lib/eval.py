import torch
import numpy as np
import open3d as o3d

from lib.metrics import pdist


def find_nn_cpu(feat0, feat1, use_open3d_feature_type=False):
  if not use_open3d_feature_type:
    feat1tree = o3d.KDTreeFlann(feat1.T)

    nn_inds = np.zeros((len(feat0),), dtype=np.int)
    for i in range(len(feat0)):
      _, idx1, _ = feat1tree.search_knn_vector_xd(feat0[i], 1)
      nn_inds[i] = idx1[0]
    return nn_inds
  else:
    pcd_tree1 = o3d.KDTreeFlann(feat1)
    nn_inds = np.zeros((feat0.num(),), dtype=np.int)
    for i in range(feat0.num()):
      _, idx1, _ = pcd_tree1.search_knn_vector_xd(feat0.data[:, i], 1)
      nn_inds[i] = idx1[0]
    return nn_inds


def find_nn_gpu(F0, F1, corr_max_n=-1, return_distance=False):
  # Too much memory if F0 or F1 large. Divide the F0
  if corr_max_n > 1:
    N = len(F0)
    C = int(np.ceil(N / corr_max_n))
    stride = corr_max_n
    dists, inds = [], []
    for i in range(C):
      dist = pdist(F0[i * stride:(i + 1) * stride], F1, dist_type='SquareL2')
      min_dist, ind = dist.min(dim=1)
      dists.append(min_dist.detach().unsqueeze(1).cpu())
      inds.append(ind.cpu())

    if C * stride < N:
      dist = pdist(F0[C * stride:], F1, dist_type='SquareL2')
      min_dist, ind = dist.min(dim=1)
      dists.append(min_dist.detach().unsqueeze(1).cpu())
      inds.append(ind.cpu())

    dists = torch.cat(dists)
    inds = torch.cat(inds)
    assert len(inds) == N
  else:
    dist = pdist(F0, F1, dist_type='SquareL2')
    min_dist, inds = dist.min(dim=1)
    dists = min_dist.detach().unsqueeze(1).cpu()
    inds = inds.cpu()
  if return_distance:
    return inds, dists
  else:
    return inds
