"""
A collection of unrefactored functions.
"""
import os
import sys
import numpy as np
import argparse
import logging
import open3d as o3d

from lib.timer import Timer, AverageMeter

from util.misc import extract_features

from model import load_model
from util.file import ensure_dir, get_folder_list, get_file_list
from util.trajectory import read_trajectory, write_trajectory
from util.pointcloud import make_open3d_point_cloud, evaluate_feature_3dmatch
from scripts.benchmark_util import do_single_pair_matching, gen_matching_pair, gather_results

import torch

import MinkowskiEngine as ME

ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    format='%(asctime)s %(message)s', datefmt='%m/%d %H:%M:%S', handlers=[ch])

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)


def extract_features_batch(model, config, source_path, target_path, voxel_size, device):

  folders = get_folder_list(source_path)
  f = open(os.path.join(target_path, "list.txt"), "w")
  timer, tmeter = Timer(), AverageMeter()
  num_feat = 0
  model.eval()

  for fo in folders:
    if 'evaluation' in fo:
      continue
    files = get_file_list(fo, ".ply")
    fo_base = os.path.basename(fo)
    f.write("%s %d\n" % (fo_base, len(files)))
    for i, fi in enumerate(files):
      # Extract features from a file
      pcd = o3d.io.read_point_cloud(fi)
      save_fn = "%s_%03d" % (fo_base, i)
      if i % 100 == 0:
        logging.info(f"{i} / {len(files)}: {save_fn}")

      timer.tic()
      xyz_down, feature = extract_features(
          model,
          xyz=np.array(pcd.points),
          rgb=None,
          normal=None,
          voxel_size=voxel_size,
          device=device,
          skip_check=True)
      t = timer.toc()
      if i > 0:
        tmeter.update(t)
        num_feat += len(xyz_down)

      np.savez_compressed(
          os.path.join(target_path, save_fn),
          points=np.array(pcd.points),
          xyz=xyz_down,
          feature=feature.detach().cpu().numpy())
      if i % 20 == 0 and i > 0:
        logging.info(
            f'Average time: {tmeter.avg}, FPS: {num_feat / tmeter.sum}, time / feat: {tmeter.sum / num_feat}, '
        )

  f.close()


def do_registration(feature_path, voxel_size):
  """
  Gather .log files produced in --target folder and run this Matlab script
  https://github.com/andyzeng/3dmatch-toolbox#geometric-registration-benchmark
  (see Geometric Registration Benchmark section in
  http://3dmatch.cs.princeton.edu/)
  """
  # List file from the extract_features_batch function
  with open(os.path.join(feature_path, "list.txt")) as f:
    sets = f.readlines()
    sets = [x.strip().split() for x in sets]
  for s in sets:
    set_name = s[0]
    pts_num = int(s[1])
    matching_pairs = gen_matching_pair(pts_num)
    results = []
    for m in matching_pairs:
      results.append(do_single_pair_matching(feature_path, set_name, m, voxel_size))
    traj = gather_results(results)
    logging.info(f"Writing the trajectory to {feature_path}/{set_name}.log")
    write_trajectory(traj, "%s.log" % (os.path.join(feature_path, set_name)))


def do_single_pair_evaluation(feature_path,
                              set_name,
                              traj,
                              voxel_size,
                              tau_1=0.1,
                              tau_2=0.05,
                              num_rand_keypoints=-1):
  trans_gth = np.linalg.inv(traj.pose)
  i = traj.metadata[0]
  j = traj.metadata[1]
  name_i = "%s_%03d" % (set_name, i)
  name_j = "%s_%03d" % (set_name, j)

  # coord and feat form a sparse tensor.
  data_i = np.load(os.path.join(feature_path, name_i + ".npz"))
  coord_i, points_i, feat_i = data_i['xyz'], data_i['points'], data_i['feature']
  data_j = np.load(os.path.join(feature_path, name_j + ".npz"))
  coord_j, points_j, feat_j = data_j['xyz'], data_j['points'], data_j['feature']

  # use the keypoints in 3DMatch
  if num_rand_keypoints > 0:
    # Randomly subsample N points
    Ni, Nj = len(points_i), len(points_j)
    inds_i = np.random.choice(Ni, min(Ni, num_rand_keypoints), replace=False)
    inds_j = np.random.choice(Nj, min(Nj, num_rand_keypoints), replace=False)

    sample_i, sample_j = points_i[inds_i], points_j[inds_j]

    key_points_i = ME.utils.fnv_hash_vec(np.floor(sample_i / voxel_size))
    key_points_j = ME.utils.fnv_hash_vec(np.floor(sample_j / voxel_size))

    key_coords_i = ME.utils.fnv_hash_vec(np.floor(coord_i / voxel_size))
    key_coords_j = ME.utils.fnv_hash_vec(np.floor(coord_j / voxel_size))

    inds_i = np.where(np.isin(key_coords_i, key_points_i))[0]
    inds_j = np.where(np.isin(key_coords_j, key_points_j))[0]

    coord_i, feat_i = coord_i[inds_i], feat_i[inds_i]
    coord_j, feat_j = coord_j[inds_j], feat_j[inds_j]

  coord_i = make_open3d_point_cloud(coord_i)
  coord_j = make_open3d_point_cloud(coord_j)

  hit_ratio = evaluate_feature_3dmatch(coord_i, coord_j, feat_i, feat_j, trans_gth,
                                       tau_1)

  # logging.info(f"Hit ratio of {name_i}, {name_j}: {hit_ratio}, {hit_ratio >= tau_2}")
  if hit_ratio >= tau_2:
    return True
  else:
    return False


def do_feature_evaluation(source_path, feature_path, voxel_size, num_rand_keypoints=-1):
  with open(os.path.join(feature_path, "list.txt")) as f:
    sets = f.readlines()
    sets = [x.strip().split() for x in sets]

  tau_1 = 0.1  # 10cm
  tau_2 = 0.05  # 5% inlier
  logging.info("%f %f" % (tau_1, tau_2))
  recall = []
  for s in sets:
    set_name = s[0]
    traj = read_trajectory(os.path.join(source_path, set_name + "_gt.log"))
    results = []
    for i in range(len(traj)):
      results.append(
          do_single_pair_evaluation(feature_path, set_name, traj[i], voxel_size, tau_1,
                                    tau_2, num_rand_keypoints))

    mean_recall = np.array(results).mean()
    std_recall = np.array(results).std()
    recall.append([set_name, mean_recall, std_recall])
    logging.info(f'{set_name}: {mean_recall} +- {std_recall}')
  for r in recall:
    logging.info("%s : %.4f" % (r[0], r[1]))
  scene_r = np.array([r[1] for r in recall])
  logging.info("average : %.4f +- %.4f" % (scene_r.mean(), scene_r.std()))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--source', default=None, type=str, help='path to 3dmatch test dataset')
  parser.add_argument(
      '--source_high_res',
      default=None,
      type=str,
      help='path to high_resolution point cloud')
  parser.add_argument(
      '--target', default=None, type=str, help='path to produce generated data')
  parser.add_argument(
      '-m',
      '--model',
      default=None,
      type=str,
      help='path to latest checkpoint (default: None)')
  parser.add_argument(
      '--voxel_size',
      default=0.05,
      type=float,
      help='voxel size to preprocess point cloud')
  parser.add_argument('--do_generate', action='store_true')
  parser.add_argument('--do_exp_feature', action='store_true')
  parser.add_argument(
      '--do_exp_registration',
      action='store_true',
      help='The target directory must contain extracted features')
  parser.add_argument('--with_cuda', action='store_true')
  parser.add_argument(
      '--num_rand_keypoints',
      type=int,
      default=5000,
      help='Number of random keypoints for each scene')

  args = parser.parse_args()

  device = torch.device('cuda' if args.with_cuda else 'cpu')

  if args.do_generate:
    assert args.model is not None
    assert args.source is not None
    assert args.target is not None

    ensure_dir(args.target)
    checkpoint = torch.load(args.model)
    config = checkpoint['config']

    num_feats = 1
    Model = load_model(config.model)
    model = Model(
        num_feats,
        config.model_n_out,
        bn_momentum=0.05,
        normalize_feature=config.normalize_feature,
        conv1_kernel_size=config.conv1_kernel_size,
        D=3)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    model = model.to(device)

    extract_features_batch(model, config, args.source, args.target, config.voxel_size,
                           device)

  if args.do_exp_feature:
    assert (args.target is not None)
    do_feature_evaluation(args.source, args.target, args.voxel_size,
                          args.num_rand_keypoints)

  if args.do_exp_registration:
    assert (args.target is not None)
    do_registration(args.target, args.voxel_size)
