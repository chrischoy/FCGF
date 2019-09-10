import MinkowskiEngine as ME


def get_norm(norm_type, num_feats, bn_momentum=0.05, D=-1):
  if norm_type == 'BN':
    return ME.MinkowskiBatchNorm(num_feats, momentum=bn_momentum)
  elif norm_type == 'IN':
    return ME.MinkowskiInstanceNorm(num_feats, dimension=D)
  else:
    raise ValueError(f'Type {norm_type}, not defined')
