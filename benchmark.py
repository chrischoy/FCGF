import os
import numpy as np
import argparse
import open3d as o3d
from urllib.request import urlretrieve
from model.resunet import ResUNetBN2C
from lib.timer import MinTimer

import torch
import MinkowskiEngine as ME

if not os.path.isfile('redkitchen-20.ply'):
  print('Downloading a mesh...')
  urlretrieve("https://node1.chrischoy.org/data/publications/fcgf/redkitchen-20.ply",
              'redkitchen-20.ply')


def benchmark(config):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  model = ResUNetBN2C(1, 16, normalize_feature=True, conv1_kernel_size=3, D=3)
  model.eval()
  model = model.to(device)

  pcd = o3d.io.read_point_cloud(config.input)
  coords = ME.utils.batched_coordinates(
      [torch.from_numpy(np.array(pcd.points)) / config.voxel_size])
  feats = torch.from_numpy(np.ones((len(coords), 1))).float()

  with torch.no_grad():
    t = MinTimer()
    for i in range(100):
      # initialization time includes copy to GPU
      t.tic()
      sinput = ME.SparseTensor(
          feats,
          coords,
          minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
          # minkowski_algorithm=ME.MinkowskiAlgorithm.MEMORY_EFFICIENT,
          device=device)
      model(sinput)
      t.toc()
    print(t.min)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '-i',
      '--input',
      default='redkitchen-20.ply',
      type=str,
      help='path to a pointcloud file')
  parser.add_argument(
      '--voxel_size',
      default=0.05,
      type=float,
      help='voxel size to preprocess point cloud')

  config = parser.parse_args()
  benchmark(config)
