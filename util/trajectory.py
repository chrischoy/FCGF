import numpy as np


class CameraPose:

  def __init__(self, meta, mat):
    self.metadata = meta
    self.pose = mat

  def __str__(self):
    return 'metadata : ' + ' '.join(map(str, self.metadata)) + '\n' + \
        "pose : " + "\n" + np.array_str(self.pose)


def read_trajectory(filename, dim=4):
  traj = []
  with open(filename, 'r') as f:
    metastr = f.readline()
    while metastr:
      metadata = list(map(int, metastr.split()))
      mat = np.zeros(shape=(dim, dim))
      for i in range(dim):
        matstr = f.readline()
        mat[i, :] = np.fromstring(matstr, dtype=float, sep=' \t')
      traj.append(CameraPose(metadata, mat))
      metastr = f.readline()
    return traj
