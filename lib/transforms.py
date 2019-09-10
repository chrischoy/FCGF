import numpy as np
import random


class Compose:

  def __init__(self, transforms):
    self.transforms = transforms

  def __call__(self, coords, feats):
    for transform in self.transforms:
      coords, feats = transform(coords, feats)
    return coords, feats


class Jitter:

  def __init__(self, mu=0, sigma=0.01):
    self.mu = mu
    self.sigma = sigma

  def __call__(self, coords, feats):
    if random.random() < 0.95:
      feats += np.random.normal(self.mu, self.sigma, (feats.shape[0], feats.shape[1]))
    return coords, feats


class ChromaticShift:

  def __init__(self, mu=0, sigma=0.1):
    self.mu = mu
    self.sigma = sigma

  def __call__(self, coords, feats):
    if random.random() < 0.95:
      feats[:, :3] += np.random.normal(self.mu, self.sigma, (1, 3))
    return coords, feats
