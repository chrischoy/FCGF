import numpy as np

import torch
import torch.functional as F


def eval_metrics(output, target):
  output = (F.sigmoid(output) > 0.5).cpu().data.numpy()
  target = target.cpu().data.numpy()
  return np.linalg.norm(output - target)


def corr_dist(est, gth, xyz0, xyz1, weight=None, max_dist=1):
  xyz0_est = xyz0 @ est[:3, :3].t() + est[:3, 3]
  xyz0_gth = xyz0 @ gth[:3, :3].t() + gth[:3, 3]
  dists = torch.clamp(torch.sqrt(((xyz0_est - xyz0_gth).pow(2)).sum(1)), max=max_dist)
  if weight is not None:
    dists = weight * dists
  return dists.mean()


def pdist(A, B, dist_type='L2'):
  if dist_type == 'L2':
    D2 = torch.sum((A.unsqueeze(1) - B.unsqueeze(0)).pow(2), 2)
    return torch.sqrt(D2 + 1e-7)
  elif dist_type == 'SquareL2':
    return torch.sum((A.unsqueeze(1) - B.unsqueeze(0)).pow(2), 2)
  else:
    raise NotImplementedError('Not implemented')


def get_loss_fn(loss):
  if loss == 'corr_dist':
    return corr_dist
  else:
    raise ValueError(f'Loss {loss}, not defined')
