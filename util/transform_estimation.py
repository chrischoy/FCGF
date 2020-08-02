import torch
import MinkowskiEngine as ME


def rot_x(x):
  out = torch.zeros((3, 3))
  c = torch.cos(x)
  s = torch.sin(x)
  out[0, 0] = 1
  out[1, 1] = c
  out[1, 2] = -s
  out[2, 1] = s
  out[2, 2] = c
  return out


def rot_y(x):
  out = torch.zeros((3, 3))
  c = torch.cos(x)
  s = torch.sin(x)
  out[0, 0] = c
  out[0, 2] = s
  out[1, 1] = 1
  out[2, 0] = -s
  out[2, 2] = c
  return out


def rot_z(x):
  out = torch.zeros((3, 3))
  c = torch.cos(x)
  s = torch.sin(x)
  out[0, 0] = c
  out[0, 1] = -s
  out[1, 0] = s
  out[1, 1] = c
  out[2, 2] = 1
  return out


def get_trans(x):
  trans = torch.eye(4)
  trans[:3, :3] = rot_z(x[2]).mm(rot_y(x[1])).mm(rot_x(x[0]))
  trans[:3, 3] = x[3:, 0]
  return trans


def update_pcd(pts, trans):
  R = trans[:3, :3]
  T = trans[:3, 3]
  # pts = R.mm(pts.t()).t() + T.unsqueeze(1).t().expand_as(pts)
  pts = torch.t(R @ torch.t(pts)) + T
  return pts


def build_linear_system(pts0, pts1, weight):
  npts0 = pts0.shape[0]
  A0 = torch.zeros((npts0, 6))
  A1 = torch.zeros((npts0, 6))
  A2 = torch.zeros((npts0, 6))
  A0[:, 1] = pts0[:, 2]
  A0[:, 2] = -pts0[:, 1]
  A0[:, 3] = 1
  A1[:, 0] = -pts0[:, 2]
  A1[:, 2] = pts0[:, 0]
  A1[:, 4] = 1
  A2[:, 0] = pts0[:, 1]
  A2[:, 1] = -pts0[:, 0]
  A2[:, 5] = 1
  ww1 = weight.repeat(3, 6)
  ww2 = weight.repeat(3, 1)
  A = ww1 * torch.cat((A0, A1, A2), 0)
  b = ww2 * torch.cat(
      (pts1[:, 0] - pts0[:, 0], pts1[:, 1] - pts0[:, 1], pts1[:, 2] - pts0[:, 2]),
      0,
  ).unsqueeze(1)
  return A, b


def solve_linear_system(A, b):
  temp = torch.inverse(A.t().mm(A))
  return temp.mm(A.t()).mm(b)


def compute_weights(pts0, pts1, par):
  return par / (torch.norm(pts0 - pts1, dim=1).unsqueeze(1) + par)


def est_quad_linear_robust(pts0, pts1, weight=None):
  # TODO: 2. residual scheduling
  pts0_curr = pts0
  trans = torch.eye(4)

  par = 1.0  # todo: need to decide
  if weight is None:
    weight = torch.ones(pts0.size()[0], 1)

  for i in range(20):
    if i > 0 and i % 5 == 0:
      par /= 2.0

    # compute weights
    A, b = build_linear_system(pts0_curr, pts1, weight)
    x = solve_linear_system(A, b)

    # TODO: early termination
    # residual = np.linalg.norm(A@x - b)
    # print(residual)

    # x = torch.empty(6, 1).uniform_(0, 1)
    trans_curr = get_trans(x)
    pts0_curr = update_pcd(pts0_curr, trans_curr)
    weight = compute_weights(pts0_curr, pts1, par)
    trans = trans_curr.mm(trans)

  return trans


def pose_estimation(model,
                    device,
                    xyz0,
                    xyz1,
                    coord0,
                    coord1,
                    feats0,
                    feats1,
                    return_corr=False):
  sinput0 = ME.SparseTensor(feats0.to(device), coordinates=coord0.to(device))
  F0 = model(sinput0).F

  sinput1 = ME.SparseTensor(feats1.to(device), coordinates=coord1.to(device))
  F1 = model(sinput1).F

  corr = F0.mm(F1.t())
  weight, inds = corr.max(dim=1)
  weight = weight.unsqueeze(1).cpu()
  xyz1_corr = xyz1[inds, :]

  trans = est_quad_linear_robust(xyz0, xyz1_corr, weight)  # let's do this later

  if return_corr:
    return trans, weight, corr
  else:
    return trans, weight
