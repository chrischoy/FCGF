# -*- coding: future_fstrings -*-
#
# Written by Chris Choy <chrischoy@ai.stanford.edu>
# Distributed under MIT License
import logging
import random
import torch
import torch.utils.data
import numpy as np
import glob
import os
from scipy.linalg import expm, norm

from util.pointcloud import get_matching_indices, make_open3d_point_cloud
import lib.transforms as t

import MinkowskiEngine as ME

import open3d as o3d

kitti_cache = {}
kitti_icp_cache = {}


def collate_pair_fn(list_data):
  xyz0, xyz1, coords0, coords1, feats0, feats1, matching_inds, trans = list(
      zip(*list_data))
  xyz_batch0, coords_batch0, feats_batch0 = [], [], []
  xyz_batch1, coords_batch1, feats_batch1 = [], [], []
  matching_inds_batch, trans_batch, len_batch = [], [], []

  batch_id = 0
  curr_start_inds = np.zeros((1, 2))
  for batch_id, _ in enumerate(coords0):
    N0 = coords0[batch_id].shape[0]
    N1 = coords1[batch_id].shape[0]

    xyz_batch0.append(torch.from_numpy(xyz0[batch_id]))
    coords_batch0.append(
        torch.cat((torch.from_numpy(
            coords0[batch_id]).int(), torch.ones(N0, 1).int() * batch_id), 1))
    feats_batch0.append(torch.from_numpy(feats0[batch_id]))

    xyz_batch1.append(torch.from_numpy(xyz1[batch_id]))
    coords_batch1.append(
        torch.cat((torch.from_numpy(
            coords1[batch_id]).int(), torch.ones(N1, 1).int() * batch_id), 1))
    feats_batch1.append(torch.from_numpy(feats1[batch_id]))

    trans_batch.append(torch.from_numpy(trans[batch_id]))

    matching_inds_batch.append(
        torch.from_numpy(np.array(matching_inds[batch_id]) + curr_start_inds))
    len_batch.append([N0, N1])

    # Move the head
    curr_start_inds[0, 0] += N0
    curr_start_inds[0, 1] += N1

  # Concatenate all lists
  xyz_batch0 = torch.cat(xyz_batch0, 0).float()
  coords_batch0 = torch.cat(coords_batch0, 0).int()
  feats_batch0 = torch.cat(feats_batch0, 0).float()
  xyz_batch1 = torch.cat(xyz_batch1, 0).float()
  coords_batch1 = torch.cat(coords_batch1, 0).int()
  feats_batch1 = torch.cat(feats_batch1, 0).float()
  trans_batch = torch.cat(trans_batch, 0).float()
  matching_inds_batch = torch.cat(matching_inds_batch, 0).int()

  return {
      'pcd0': xyz_batch0,
      'pcd1': xyz_batch1,
      'sinput0_C': coords_batch0,
      'sinput0_F': feats_batch0,
      'sinput1_C': coords_batch1,
      'sinput1_F': feats_batch1,
      'correspondences': matching_inds_batch,
      'T_gt': trans_batch,
      'len_batch': len_batch
  }


# Rotation matrix along axis with angle theta
def M(axis, theta):
  return expm(np.cross(np.eye(3), axis / norm(axis) * theta))


def sample_random_trans(pcd, randg, rotation_range=360):
  T = np.eye(4)
  R = M(randg.rand(3) - 0.5, rotation_range * np.pi / 180.0 * (randg.rand(1) - 0.5))
  T[:3, :3] = R
  T[:3, 3] = R.dot(-np.mean(pcd, axis=0))
  return T


class PairDataset(torch.utils.data.Dataset):
  AUGMENT = None

  def __init__(self,
               phase,
               transform=None,
               random_rotation=True,
               random_scale=True,
               manual_seed=False,
               config=None):
    self.phase = phase
    self.files = []
    self.data_objects = []
    self.transform = transform
    self.voxel_size = config.voxel_size
    self.matching_search_voxel_size = \
        config.voxel_size * config.positive_pair_search_voxel_size_multiplier

    self.random_scale = random_scale
    self.min_scale = config.min_scale
    self.max_scale = config.max_scale
    self.random_rotation = random_rotation
    self.rotation_range = config.rotation_range
    self.randg = np.random.RandomState()
    if manual_seed:
      self.reset_seed()

  def reset_seed(self, seed=0):
    logging.info(f"Resetting the data loader seed to {seed}")
    self.randg.seed(seed)

  def apply_transform(self, pts, trans):
    R = trans[:3, :3]
    T = trans[:3, 3]
    pts = pts @ R.T + T
    return pts

  def __len__(self):
    return len(self.files)


class IndoorPairDataset(PairDataset):
  OVERLAP_RATIO = None
  AUGMENT = None

  def __init__(self,
               phase,
               transform=None,
               random_rotation=True,
               random_scale=True,
               manual_seed=False,
               config=None):
    PairDataset.__init__(self, phase, transform, random_rotation, random_scale,
                         manual_seed, config)
    self.root = root = config.data_dir_10mm if config.use_10mm else config.data_dir_25mm
    logging.info(f"Loading the subset {phase} from {root}")

    subset_names = open(self.DATA_FILES[phase]).read().split()
    for name in subset_names:
      fname = name + "*%.2f.txt" % self.OVERLAP_RATIO
      fnames_txt = glob.glob(root + "/" + fname)
      assert len(fnames_txt) > 0, f"Make sure that the path {root} has data {fname}"
      for fname_txt in fnames_txt:
        with open(fname_txt) as f:
          content = f.readlines()
        fnames = [x.strip().split() for x in content]
        for fname in fnames:
          self.files.append([fname[0], fname[1]])

  def __getitem__(self, idx):
    file0 = os.path.join(self.root, self.files[idx][0])
    file1 = os.path.join(self.root, self.files[idx][1])
    data0 = np.load(file0)
    data1 = np.load(file1)
    xyz0 = data0["pcd"]
    xyz1 = data1["pcd"]
    color0 = data0["color"]
    color1 = data1["color"]
    matching_search_voxel_size = self.matching_search_voxel_size

    if self.random_scale and random.random() < 0.95:
      scale = self.min_scale + \
          (self.max_scale - self.min_scale) * random.random()
      matching_search_voxel_size *= scale
      xyz0 = scale * xyz0
      xyz1 = scale * xyz1

    if self.random_rotation:
      T0 = sample_random_trans(xyz0, self.randg, self.rotation_range)
      T1 = sample_random_trans(xyz1, self.randg, self.rotation_range)
      trans = T1 @ np.linalg.inv(T0)

      xyz0 = self.apply_transform(xyz0, T0)
      xyz1 = self.apply_transform(xyz1, T1)
    else:
      trans = np.identity(4)

    # Voxelization
    sel0 = ME.utils.sparse_quantize(xyz0 / self.voxel_size, return_index=True)
    sel1 = ME.utils.sparse_quantize(xyz1 / self.voxel_size, return_index=True)

    # Make point clouds using voxelized points
    pcd0 = make_open3d_point_cloud(xyz0)
    pcd1 = make_open3d_point_cloud(xyz1)

    # Select features and points using the returned voxelized indices
    pcd0.colors = o3d.Vector3dVector(color0[sel0])
    pcd1.colors = o3d.Vector3dVector(color1[sel1])
    pcd0.points = o3d.Vector3dVector(np.array(pcd0.points)[sel0])
    pcd1.points = o3d.Vector3dVector(np.array(pcd1.points)[sel1])
    # Get matches
    matches = get_matching_indices(pcd0, pcd1, trans, matching_search_voxel_size)

    # Get features
    npts0 = len(pcd0.colors)
    npts1 = len(pcd1.colors)

    feats_train0, feats_train1 = [], []

    feats_train0.append(np.ones((npts0, 1)))
    feats_train1.append(np.ones((npts1, 1)))

    feats0 = np.hstack(feats_train0)
    feats1 = np.hstack(feats_train1)

    # Get coords
    xyz0 = np.array(pcd0.points)
    xyz1 = np.array(pcd1.points)

    coords0 = np.floor(xyz0 / self.voxel_size)
    coords1 = np.floor(xyz1 / self.voxel_size)

    if self.transform:
      coords0, feats0 = self.transform(coords0, feats0)
      coords1, feats1 = self.transform(coords1, feats1)

    return (xyz0, xyz1, coords0, coords1, feats0, feats1, matches, trans)


class KITTIPairDataset(PairDataset):
  AUGMENT = None
  DATA_FILES = {
      'train': './config/train_kitti.txt',
      'val': './config/val_kitti.txt',
      'test': './config/test_kitti.txt'
  }
  TEST_RANDOM_ROTATION = False
  IS_ODOMETRY = True

  def __init__(self,
               phase,
               transform=None,
               random_rotation=True,
               random_scale=True,
               manual_seed=False,
               config=None):
    # For evaluation, use the odometry dataset training following the 3DFeat eval method
    if self.IS_ODOMETRY:
      self.root = root = config.kitti_root + '/dataset'
      random_rotation = self.TEST_RANDOM_ROTATION
    else:
      self.date = config.kitti_date
      self.root = root = os.path.join(config.kitti_root, self.date)

    self.icp_path = config.icp_cache_path
    PairDataset.__init__(self, phase, transform, random_rotation, random_scale,
                         manual_seed, config)

    logging.info(f"Loading the subset {phase} from {root}")
    # Use the kitti root
    self.max_time_diff = max_time_diff = config.kitti_max_time_diff

    subset_names = open(self.DATA_FILES[phase]).read().split()
    for dirname in subset_names:
      drive_id = int(dirname)
      inames = self.get_all_scan_ids(drive_id)
      for start_time in inames:
        for time_diff in range(2, max_time_diff):
          pair_time = time_diff + start_time
          if pair_time in inames:
            self.files.append((drive_id, start_time, pair_time))

  def get_all_scan_ids(self, drive_id):
    if self.IS_ODOMETRY:
      fnames = glob.glob(self.root + '/sequences/%02d/velodyne/*.bin' % drive_id)
    else:
      fnames = glob.glob(self.root + '/' + self.date +
                         '_drive_%04d_sync/velodyne_points/data/*.bin' % drive_id)
    assert len(
        fnames) > 0, f"Make sure that the path {self.root} has drive id: {drive_id}"
    inames = [int(os.path.split(fname)[-1][:-4]) for fname in fnames]
    return inames

  @property
  def velo2cam(self):
    try:
      velo2cam = self._velo2cam
    except AttributeError:
      R = np.array([
          7.533745e-03, -9.999714e-01, -6.166020e-04, 1.480249e-02, 7.280733e-04,
          -9.998902e-01, 9.998621e-01, 7.523790e-03, 1.480755e-02
      ]).reshape(3, 3)
      T = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01]).reshape(3, 1)
      velo2cam = np.hstack([R, T])
      self._velo2cam = np.vstack((velo2cam, [0, 0, 0, 1])).T
    return self._velo2cam

  def get_video_odometry(self, drive, indices=None, ext='.txt', return_all=False):
    if self.IS_ODOMETRY:
      data_path = self.root + '/poses/%02d.txt' % drive
      if data_path not in kitti_cache:
        kitti_cache[data_path] = np.genfromtxt(data_path)
      if return_all:
        return kitti_cache[data_path]
      else:
        return kitti_cache[data_path][indices]
    else:
      data_path = self.root + '/' + self.date + '_drive_%04d_sync/oxts/data' % drive
      odometry = []
      if indices is None:
        fnames = glob.glob(self.root + '/' + self.date +
                           '_drive_%04d_sync/velodyne_points/data/*.bin' % drive)
        indices = sorted([int(os.path.split(fname)[-1][:-4]) for fname in fnames])

      for index in indices:
        filename = os.path.join(data_path, '%010d%s' % (index, ext))
        if filename not in kitti_cache:
          kitti_cache[filename] = np.genfromtxt(filename)
        odometry.append(kitti_cache[filename])

      odometry = np.array(odometry)
      return odometry

  def odometry_to_positions(self, odometry):
    if self.IS_ODOMETRY:
      T_w_cam0 = odometry.reshape(3, 4)
      T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
      return T_w_cam0
    else:
      lat, lon, alt, roll, pitch, yaw = odometry.T[:6]

      R = 6378137  # Earth's radius in metres

      # convert to metres
      lat, lon = np.deg2rad(lat), np.deg2rad(lon)
      mx = R * lon * np.cos(lat)
      my = R * lat

      times = odometry.T[-1]
      return np.vstack([mx, my, alt, roll, pitch, yaw, times]).T

  def rot3d(self, axis, angle):
    ei = np.ones(3, dtype='bool')
    ei[axis] = 0
    i = np.nonzero(ei)[0]
    m = np.eye(3)
    c, s = np.cos(angle), np.sin(angle)
    m[i[0], i[0]] = c
    m[i[0], i[1]] = -s
    m[i[1], i[0]] = s
    m[i[1], i[1]] = c
    return m

  def pos_transform(self, pos):
    x, y, z, rx, ry, rz, _ = pos[0]
    RT = np.eye(4)
    RT[:3, :3] = np.dot(np.dot(self.rot3d(0, rx), self.rot3d(1, ry)), self.rot3d(2, rz))
    RT[:3, 3] = [x, y, z]
    return RT

  def get_position_transform(self, pos0, pos1, invert=False):
    T0 = self.pos_transform(pos0)
    T1 = self.pos_transform(pos1)
    return (np.dot(T1, np.linalg.inv(T0)).T if not invert else np.dot(
        np.linalg.inv(T1), T0).T)

  def _get_velodyne_fn(self, drive, t):
    if self.IS_ODOMETRY:
      fname = self.root + '/sequences/%02d/velodyne/%06d.bin' % (drive, t)
    else:
      fname = self.root + \
          '/' + self.date + '_drive_%04d_sync/velodyne_points/data/%010d.bin' % (
              drive, t)
    return fname

  def __getitem__(self, idx):
    drive = self.files[idx][0]
    t0, t1 = self.files[idx][1], self.files[idx][2]
    all_odometry = self.get_video_odometry(drive, [t0, t1])
    positions = [self.odometry_to_positions(odometry) for odometry in all_odometry]
    fname0 = self._get_velodyne_fn(drive, t0)
    fname1 = self._get_velodyne_fn(drive, t1)

    # XYZ and reflectance
    xyzr0 = np.fromfile(fname0, dtype=np.float32).reshape(-1, 4)
    xyzr1 = np.fromfile(fname1, dtype=np.float32).reshape(-1, 4)

    xyz0 = xyzr0[:, :3]
    xyz1 = xyzr1[:, :3]

    coords0 = (xyz0 - xyz0.min(0)) / 0.05
    coords1 = (xyz1 - xyz1.min(0)) / 0.05

    sel0 = ME.utils.sparse_quantize(coords0, return_index=True)
    sel1 = ME.utils.sparse_quantize(coords1, return_index=True)

    xyz0 = xyz0[sel0]
    xyz1 = xyz1[sel1]

    # r0 = xyzr0[:, -1].reshape(-1, 1)
    # r1 = xyzr1[:, -1].reshape(-1, 1)

    # pcd0 = make_open3d_point_cloud(xyz0_t, 0.7 * np.ones((len(xyz0), 3)))
    # pcd1 = make_open3d_point_cloud(xyz1, 0.3 * np.ones((len(xyz1), 3)))

    key = '%d_%d_%d' % (drive, t0, t1)
    filename = self.icp_path + '/' + key + '.npy'
    if key not in kitti_icp_cache:
      if not os.path.exists(filename):
        if self.IS_ODOMETRY:
          M = (self.velo2cam @ positions[0].T @ np.linalg.inv(positions[1].T)
               @ np.linalg.inv(self.velo2cam)).T
        else:
          M = self.get_position_transform(positions[0], positions[1], invert=True).T
        xyz0_t = self.apply_transform(xyz0, M)
        pcd0 = make_open3d_point_cloud(xyz0_t)
        pcd1 = make_open3d_point_cloud(xyz1)
        reg = o3d.registration_icp(pcd0, pcd1, 0.2, np.eye(4),
                                   o3d.TransformationEstimationPointToPoint(),
                                   o3d.ICPConvergenceCriteria(max_iteration=200))
        pcd0.transform(reg.transformation)
        # pcd0.transform(M2) or self.apply_transform(xyz0, M2)
        M2 = M @ reg.transformation
        # o3d.draw_geometries([pcd0, pcd1])
        # write to a file
        np.save(filename, M2)
      else:
        M2 = np.load(filename)
      kitti_icp_cache[key] = M2
    else:
      M2 = kitti_icp_cache[key]

    if self.random_rotation:
      T0 = sample_random_trans(xyz0, self.randg, np.pi / 4)
      T1 = sample_random_trans(xyz1, self.randg, np.pi / 4)
      trans = T1 @ M2 @ np.linalg.inv(T0)

      xyz0 = self.apply_transform(xyz0, T0)
      xyz1 = self.apply_transform(xyz1, T1)
    else:
      trans = M2

    matching_search_voxel_size = self.matching_search_voxel_size
    if self.random_scale and random.random() < 0.95:
      scale = self.min_scale + \
          (self.max_scale - self.min_scale) * random.random()
      matching_search_voxel_size *= scale
      xyz0 = scale * xyz0
      xyz1 = scale * xyz1

    # Voxelization
    coords0 = np.floor(xyz0 / self.voxel_size)
    coords1 = np.floor(xyz1 / self.voxel_size)
    sel0 = ME.utils.sparse_quantize(coords0, return_index=True)
    sel1 = ME.utils.sparse_quantize(coords1, return_index=True)
    coords0, coords1 = coords0[sel0], coords1[sel1]
    # r0, r1 = r0[sel0], r1[sel1]

    pcd0 = make_open3d_point_cloud(xyz0)
    pcd1 = make_open3d_point_cloud(xyz1)

    # Select features and points using the returned voxelized indices
    pcd0.points = o3d.Vector3dVector(np.array(pcd0.points)[sel0])
    pcd1.points = o3d.Vector3dVector(np.array(pcd1.points)[sel1])

    # Get matches
    matches = get_matching_indices(pcd0, pcd1, trans, matching_search_voxel_size)
    if len(matches) < 1000:
      raise ValueError(f"{drive}, {t0}, {t1}")

    feats_train0, feats_train1 = [], []

    feats_train0.append(np.ones((len(sel0), 1)))
    feats_train1.append(np.ones((len(sel1), 1)))

    feats0 = np.hstack(feats_train0)
    feats1 = np.hstack(feats_train1)

    # Get coords
    xyz0 = np.array(pcd0.points)
    xyz1 = np.array(pcd1.points)

    if self.transform:
      coords0, feats0 = self.transform(coords0, feats0)
      coords1, feats1 = self.transform(coords1, feats1)

    return (xyz0, xyz1, coords0, coords1, feats0, feats1, matches, trans)


class KITTINMPairDataset(KITTIPairDataset):
  r"""
  Generate KITTI pairs within N meter distance
  """
  MAX_TIME_DIFF = 3

  def __init__(self,
               phase,
               transform=None,
               random_rotation=True,
               random_scale=True,
               manual_seed=False,
               config=None):
    if self.IS_ODOMETRY:
      self.root = root = config.kitti_root + '/dataset'
      random_rotation = self.TEST_RANDOM_ROTATION
    else:
      self.date = config.kitti_date
      self.root = root = os.path.join(config.kitti_root, self.date)
    PairDataset.__init__(self, phase, transform, random_rotation, random_scale,
                         manual_seed, config)
    self.icp_path = config.icp_cache_path

    logging.info(f"Loading the subset {phase} from {root}")
    # Use the kitti root
    if phase == 'train':
      max_time_diff = self.MAX_TIME_DIFF
    else:
      max_time_diff = -1

    subset_names = open(self.DATA_FILES[phase]).read().split()
    if self.IS_ODOMETRY:
      for dirname in subset_names:
        drive_id = int(dirname)
        fnames = glob.glob(root + '/sequences/%02d/velodyne/*.bin' % drive_id)
        assert len(fnames) > 0, f"Make sure that the path {root} has data {dirname}"
        inames = sorted([int(os.path.split(fname)[-1][:-4]) for fname in fnames])

        all_odo = self.get_video_odometry(drive_id, return_all=True)
        all_pos = np.array([self.odometry_to_positions(odo) for odo in all_odo])
        Ts = all_pos[:, :3, 3]
        pdist = (Ts.reshape(1, -1, 3) - Ts.reshape(-1, 1, 3))**2
        pdist = np.sqrt(pdist.sum(-1))
        more_than_10 = pdist > 10
        curr_time = inames[0]
        while curr_time in inames:
          # Find the min index
          next_time = np.where(more_than_10[curr_time][curr_time:curr_time + 100])[0]
          if len(next_time) == 0:
            curr_time += 1
          else:
            # Follow https://github.com/yewzijian/3DFeatNet/blob/master/scripts_data_processing/kitti/process_kitti_data.m#L44
            next_time = next_time[0] + curr_time - 1

          if next_time in inames:
            self.files.append((drive_id, curr_time, next_time))
            curr_time = next_time + 1
    else:
      for dirname in subset_names:
        drive_id = int(dirname)
        fnames = glob.glob(root + '/' + self.date +
                           '_drive_%04d_sync/velodyne_points/data/*.bin' % drive_id)
        assert len(fnames) > 0, f"Make sure that the path {root} has data {dirname}"
        inames = sorted([int(os.path.split(fname)[-1][:-4]) for fname in fnames])

        all_odo = self.get_video_odometry(drive_id, return_all=True)
        all_pos = np.array([self.odometry_to_positions(odo) for odo in all_odo])
        Ts = all_pos[:, 0, :3]

        pdist = (Ts.reshape(1, -1, 3) - Ts.reshape(-1, 1, 3))**2
        pdist = np.sqrt(pdist.sum(-1))

        for start_time in inames:
          pair_time = np.where(pdist[start_time][start_time:start_time + 100] > 10)[0]
          if len(pair_time) == 0:
            continue
          else:
            pair_time = pair_time[0] + start_time

          if pair_time in inames:
            self.files.append((drive_id, start_time, pair_time))

          if max_time_diff > 0:
            for diff in range(1, max_time_diff):
              if pair_time + diff in inames:
                self.files.append((drive_id, start_time, pair_time + diff))

    if self.IS_ODOMETRY:
      # Remove problematic sequence
      for item in [
          (8, 15, 58),
      ]:
        if item in self.files:
          self.files.pop(self.files.index(item))


class ThreeDMatchPairDataset(IndoorPairDataset):
  OVERLAP_RATIO = 0.3
  DATA_FILES = {
      'train': './config/train_3dmatch.txt',
      'val': './config/val_3dmatch.txt',
      'test': './config/test_3dmatch.txt'
  }


ALL_DATASETS = [ThreeDMatchPairDataset, KITTIPairDataset, KITTINMPairDataset]
dataset_str_mapping = {d.__name__: d for d in ALL_DATASETS}


def make_data_loader(config, phase, batch_size, num_threads=0, shuffle=None):
  assert phase in ['train', 'trainval', 'val', 'test']
  if shuffle is None:
    shuffle = phase != 'test'

  if config.dataset not in dataset_str_mapping.keys():
    logging.error(f'Dataset {config.dataset}, does not exists in ' +
                  ', '.join(dataset_str_mapping.keys()))

  Dataset = dataset_str_mapping[config.dataset]

  use_random_scale = False
  use_random_rotation = False
  transforms = []
  if phase in ['train', 'trainval']:
    use_random_rotation = config.use_random_rotation
    use_random_scale = config.use_random_scale
    transforms += [t.Jitter()]

  dset = Dataset(
      phase,
      transform=t.Compose(transforms),
      random_scale=use_random_scale,
      random_rotation=use_random_rotation,
      config=config)

  loader = torch.utils.data.DataLoader(
      dset,
      batch_size=batch_size,
      shuffle=shuffle,
      num_workers=num_threads,
      collate_fn=collate_pair_fn,
      pin_memory=False,
      drop_last=True)

  return loader
