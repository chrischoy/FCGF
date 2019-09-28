import open3d as o3d  # prevent loading error

import sys
import logging
import json
import argparse
import numpy as np
from easydict import EasyDict as edict

import torch
from model import load_model

from lib.data_loaders import make_data_loader
from util.pointcloud import make_open3d_point_cloud, make_open3d_feature
from lib.loss import corr_dist
from lib.timer import AverageMeter, Timer

import MinkowskiEngine as ME

ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    format='%(asctime)s %(message)s', datefmt='%m/%d %H:%M:%S', handlers=[ch])


def main(config):
  test_loader = make_data_loader(
      config, config.test_phase, 1, num_threads=config.test_num_thread, shuffle=True)

  num_feats = 1

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  Model = load_model(config.model)
  model = Model(num_feats, config.model_n_out, config=config)
  checkpoint = torch.load(config.save_dir + '/checkpoint.pth')
  model.load_state_dict(checkpoint['state_dict'])
  model = model.to(device)
  model.eval()

  success_meter, loss_meter, rte_meter, rre_meter = AverageMeter(), AverageMeter(
  ), AverageMeter(), AverageMeter()
  data_timer, feat_timer, reg_timer = Timer(), Timer(), Timer()

  test_iter = test_loader.__iter__()
  N = len(test_iter)
  n_gpu_failures = 0

  # downsample_voxel_size = 2 * config.voxel_size

  for i in range(len(test_iter)):

    data_timer.tic()
    try:
      xyz0, xyz1, coord0, coord1, feats0, feats1, matching01, T_gth, len_01 = test_iter.next(
      )
    except ValueError:
      n_gpu_failures += 1
      logging.info(f"# Erroneous GPU Pair {n_gpu_failures}")
      continue
    data_timer.toc()

    xyz0np, xyz1np = xyz0.numpy(), xyz1.numpy()

    pcd0 = make_open3d_point_cloud(xyz0np)
    pcd1 = make_open3d_point_cloud(xyz1np)

    with torch.no_grad():
      feat_timer.tic()
      sinput0 = ME.SparseTensor(feats0, coords=coord0).to(device)
      F0 = model(sinput0).F.detach()
      sinput1 = ME.SparseTensor(feats1, coords=coord1).to(device)
      F1 = model(sinput1).F.detach()
      feat_timer.toc()

    feat0 = make_open3d_feature(F0, 32, F0.shape[0])
    feat1 = make_open3d_feature(F1, 32, F1.shape[0])

    reg_timer.tic()
    distance_threshold = config.voxel_size * 1.0
    ransac_result = o3d.registration_ransac_based_on_feature_matching(
        pcd0, pcd1, feat0, feat1, distance_threshold,
        o3d.TransformationEstimationPointToPoint(False), 4, [
            o3d.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ], o3d.RANSACConvergenceCriteria(4000000, 10000))
    T_ransac = torch.from_numpy(ransac_result.transformation.astype(np.float32))
    reg_timer.toc()

    loss_ransac = corr_dist(T_ransac, T_gth, xyz0, xyz1, weight=None, max_dist=1)

    # Translation error
    rte = np.linalg.norm(T_ransac[:3, 3] - T_gth[:3, 3])
    rre = np.arccos((np.trace(T_ransac[:3, :3].t() @ T_gth[:3, :3]) - 1) / 2)

    # Check if the ransac was successful. successful if rte < 2m and rre < 5◦
    # http://openaccess.thecvf.com/content_ECCV_2018/papers/Zi_Jian_Yew_3DFeat-Net_Weakly_Supervised_ECCV_2018_paper.pdf
    if rte < 2:
      rte_meter.update(rte)

    if not np.isnan(rre) and rre < np.pi / 180 * 5:
      rre_meter.update(rre)

    if rte < 2 and not np.isnan(rre) and rre < np.pi / 180 * 5:
      success_meter.update(1)
    else:
      success_meter.update(0)
      logging.info(f"Failed with RTE: {rte}, RRE: {rre}")

    loss_meter.update(loss_ransac)

    if i % 10 == 0:
      logging.info(
          f"{i} / {N}: Data time: {data_timer.avg}, Feat time: {feat_timer.avg}," +
          f" Reg time: {reg_timer.avg}, Loss: {loss_meter.avg}, RTE: {rte_meter.avg}," +
          f" RRE: {rre_meter.avg}, Success: {success_meter.sum} / {success_meter.count}" +
          f" ({success_meter.avg * 100} %)"
      )
      data_timer.reset()
      feat_timer.reset()
      reg_timer.reset()

  logging.info(
      f"Total loss: {loss_meter.avg}, RTE: {rte_meter.avg}, var: {rte_meter.var}," +
      f" RRE: {rre_meter.avg}, var: {rre_meter.var}, Success: {success_meter.sum} " +
      f"/ {success_meter.count} ({success_meter.avg * 100} %)"
  )


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--save_dir', default=None, type=str)
  parser.add_argument('--test_phase', default='test', type=str)
  parser.add_argument('--test_num_thread', default=5, type=int)
  parser.add_argument('--kitti_root', type=str, default="/data/kitti/")
  args = parser.parse_args()

  config = json.load(open(args.save_dir + '/config.json', 'r'))
  config = edict(config)
  config.save_dir = args.save_dir
  config.test_phase = args.test_phase
  config.kitti_root = args.kitti_root
  config.kitti_odometry_root = args.kitti_root + '/dataset'
  config.icp_cache_path = args.kitti_root + '/icp'
  config.test_num_thread = args.test_num_thread

  main(config)
