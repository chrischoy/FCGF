# -*- coding: future_fstrings -*-
import torch
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MEF
from model.common import get_norm


class SimpleNet(ME.MinkowskiNetwork):
  NORM_TYPE = None
  CHANNELS = [None, 32, 64, 128]
  TR_CHANNELS = [None, 32, 32, 64]

  # To use the model, must call initialize_coords before forward pass.
  # Once data is processed, call clear to reset the model before calling initialize_coords
  def __init__(self,
               in_channels=3,
               out_channels=32,
               bn_momentum=0.1,
               normalize_feature=None,
               conv1_kernel_size=None,
               D=3):
    super(SimpleNet, self).__init__(D)
    NORM_TYPE = self.NORM_TYPE
    CHANNELS = self.CHANNELS
    TR_CHANNELS = self.TR_CHANNELS
    self.normalize_feature = normalize_feature
    self.conv1 = ME.MinkowskiConvolution(
        in_channels=in_channels,
        out_channels=CHANNELS[1],
        kernel_size=conv1_kernel_size,
        stride=1,
        dilation=1,
        has_bias=False,
        dimension=D)
    self.norm1 = get_norm(NORM_TYPE, CHANNELS[1], bn_momentum=bn_momentum, D=D)

    self.conv2 = ME.MinkowskiConvolution(
        in_channels=CHANNELS[1],
        out_channels=CHANNELS[2],
        kernel_size=3,
        stride=2,
        dilation=1,
        has_bias=False,
        dimension=D)
    self.norm2 = get_norm(NORM_TYPE, CHANNELS[2], bn_momentum=bn_momentum, D=D)

    self.conv3 = ME.MinkowskiConvolution(
        in_channels=CHANNELS[2],
        out_channels=CHANNELS[3],
        kernel_size=3,
        stride=2,
        dilation=1,
        has_bias=False,
        dimension=D)
    self.norm3 = get_norm(NORM_TYPE, CHANNELS[3], bn_momentum=bn_momentum, D=D)

    self.conv3_tr = ME.MinkowskiConvolutionTranspose(
        in_channels=CHANNELS[3],
        out_channels=TR_CHANNELS[3],
        kernel_size=3,
        stride=2,
        dilation=1,
        has_bias=False,
        dimension=D)
    self.norm3_tr = get_norm(NORM_TYPE, TR_CHANNELS[3], bn_momentum=bn_momentum, D=D)

    self.conv2_tr = ME.MinkowskiConvolutionTranspose(
        in_channels=CHANNELS[2] + TR_CHANNELS[3],
        out_channels=TR_CHANNELS[2],
        kernel_size=3,
        stride=2,
        dilation=1,
        has_bias=False,
        dimension=D)
    self.norm2_tr = get_norm(NORM_TYPE, TR_CHANNELS[2], bn_momentum=bn_momentum, D=D)

    self.conv1_tr = ME.MinkowskiConvolution(
        in_channels=CHANNELS[1] + TR_CHANNELS[2],
        out_channels=TR_CHANNELS[1],
        kernel_size=3,
        stride=1,
        dilation=1,
        has_bias=False,
        dimension=D)
    self.norm1_tr = get_norm(NORM_TYPE, TR_CHANNELS[1], bn_momentum=bn_momentum, D=D)

    self.final = ME.MinkowskiConvolution(
        in_channels=TR_CHANNELS[1],
        out_channels=out_channels,
        kernel_size=1,
        stride=1,
        dilation=1,
        has_bias=True,
        dimension=D)

  def forward(self, x):
    out_s1 = self.conv1(x)
    out_s1 = self.norm1(out_s1)
    out = MEF.relu(out_s1)

    out_s2 = self.conv2(out)
    out_s2 = self.norm2(out_s2)
    out = MEF.relu(out_s2)

    out_s4 = self.conv3(out)
    out_s4 = self.norm3(out_s4)
    out = MEF.relu(out_s4)

    out = self.conv3_tr(out)
    out = self.norm3_tr(out)
    out_s2_tr = MEF.relu(out)

    out = ME.cat((out_s2_tr, out_s2))

    out = self.conv2_tr(out)
    out = self.norm2_tr(out)
    out_s1_tr = MEF.relu(out)

    out = ME.cat((out_s1_tr, out_s1))
    out = self.conv1_tr(out)
    out = self.norm1_tr(out)
    out = MEF.relu(out)

    out = self.final(out)

    if self.normalize_feature:
      return ME.SparseTensor(
          out.F / torch.norm(out.F, p=2, dim=1, keepdim=True),
          coords_key=out.coords_key,
          coords_manager=out.coords_man)
    else:
      return out


class SimpleNetIN(SimpleNet):
  NORM_TYPE = 'IN'


class SimpleNetBN(SimpleNet):
  NORM_TYPE = 'BN'


class SimpleNetBNE(SimpleNetBN):
  CHANNELS = [None, 16, 32, 32]
  TR_CHANNELS = [None, 16, 16, 32]


class SimpleNetINE(SimpleNetBNE):
  NORM_TYPE = 'IN'


class SimpleNet2(ME.MinkowskiNetwork):
  NORM_TYPE = None
  CHANNELS = [None, 32, 64, 128, 256]
  TR_CHANNELS = [None, 32, 32, 64, 64]

  # To use the model, must call initialize_coords before forward pass.
  # Once data is processed, call clear to reset the model before calling initialize_coords
  def __init__(self, in_channels=3, out_channels=32, bn_momentum=0.1, D=3, config=None):
    ME.MinkowskiNetwork.__init__(self, D)
    NORM_TYPE = self.NORM_TYPE
    bn_momentum = config.bn_momentum
    CHANNELS = self.CHANNELS
    TR_CHANNELS = self.TR_CHANNELS
    self.normalize_feature = config.normalize_feature
    self.conv1 = ME.MinkowskiConvolution(
        in_channels=in_channels,
        out_channels=CHANNELS[1],
        kernel_size=config.conv1_kernel_size,
        stride=1,
        dilation=1,
        has_bias=False,
        dimension=D)
    self.norm1 = get_norm(NORM_TYPE, CHANNELS[1], bn_momentum=bn_momentum, D=D)

    self.conv2 = ME.MinkowskiConvolution(
        in_channels=CHANNELS[1],
        out_channels=CHANNELS[2],
        kernel_size=3,
        stride=2,
        dilation=1,
        has_bias=False,
        dimension=D)
    self.norm2 = get_norm(NORM_TYPE, CHANNELS[2], bn_momentum=bn_momentum, D=D)

    self.conv3 = ME.MinkowskiConvolution(
        in_channels=CHANNELS[2],
        out_channels=CHANNELS[3],
        kernel_size=3,
        stride=2,
        dilation=1,
        has_bias=False,
        dimension=D)
    self.norm3 = get_norm(NORM_TYPE, CHANNELS[3], bn_momentum=bn_momentum, D=D)

    self.conv4 = ME.MinkowskiConvolution(
        in_channels=CHANNELS[3],
        out_channels=CHANNELS[4],
        kernel_size=3,
        stride=2,
        dilation=1,
        has_bias=False,
        dimension=D)
    self.norm4 = get_norm(NORM_TYPE, CHANNELS[4], bn_momentum=bn_momentum, D=D)

    self.conv4_tr = ME.MinkowskiConvolutionTranspose(
        in_channels=CHANNELS[4],
        out_channels=TR_CHANNELS[4],
        kernel_size=3,
        stride=2,
        dilation=1,
        has_bias=False,
        dimension=D)
    self.norm4_tr = get_norm(NORM_TYPE, TR_CHANNELS[4], bn_momentum=bn_momentum, D=D)

    self.conv3_tr = ME.MinkowskiConvolutionTranspose(
        in_channels=CHANNELS[3] + TR_CHANNELS[4],
        out_channels=TR_CHANNELS[3],
        kernel_size=3,
        stride=2,
        dilation=1,
        has_bias=False,
        dimension=D)
    self.norm3_tr = get_norm(NORM_TYPE, TR_CHANNELS[3], bn_momentum=bn_momentum, D=D)

    self.conv2_tr = ME.MinkowskiConvolutionTranspose(
        in_channels=CHANNELS[2] + TR_CHANNELS[3],
        out_channels=TR_CHANNELS[2],
        kernel_size=3,
        stride=2,
        dilation=1,
        has_bias=False,
        dimension=D)
    self.norm2_tr = get_norm(NORM_TYPE, TR_CHANNELS[2], bn_momentum=bn_momentum, D=D)

    self.conv1_tr = ME.MinkowskiConvolution(
        in_channels=CHANNELS[1] + TR_CHANNELS[2],
        out_channels=TR_CHANNELS[1],
        kernel_size=3,
        stride=1,
        dilation=1,
        has_bias=False,
        dimension=D)
    self.norm1_tr = get_norm(NORM_TYPE, TR_CHANNELS[1], bn_momentum=bn_momentum, D=D)

    self.final = ME.MinkowskiConvolution(
        in_channels=TR_CHANNELS[1],
        out_channels=out_channels,
        kernel_size=1,
        stride=1,
        dilation=1,
        has_bias=True,
        dimension=D)

  def forward(self, x):
    out_s1 = self.conv1(x)
    out_s1 = self.norm1(out_s1)
    out = MEF.relu(out_s1)

    out_s2 = self.conv2(out)
    out_s2 = self.norm2(out_s2)
    out = MEF.relu(out_s2)

    out_s4 = self.conv3(out)
    out_s4 = self.norm3(out_s4)
    out = MEF.relu(out_s4)

    out_s8 = self.conv4(out)
    out_s8 = self.norm4(out_s8)
    out = MEF.relu(out_s8)

    out = self.conv4_tr(out)
    out = self.norm4_tr(out)
    out_s4_tr = MEF.relu(out)

    out = ME.cat((out_s4_tr, out_s4))

    out = self.conv3_tr(out)
    out = self.norm3_tr(out)
    out_s2_tr = MEF.relu(out)

    out = ME.cat((out_s2_tr, out_s2))

    out = self.conv2_tr(out)
    out = self.norm2_tr(out)
    out_s1_tr = MEF.relu(out)

    out = ME.cat((out_s1_tr, out_s1))
    out = self.conv1_tr(out)
    out = self.norm1_tr(out)
    out = MEF.relu(out)

    out = self.final(out)

    if self.normalize_feature:
      return ME.SparseTensor(
          out.F / torch.norm(out.F, p=2, dim=1, keepdim=True),
          coords_key=out.coords_key,
          coords_manager=out.coords_man)
    else:
      return out


class SimpleNetIN2(SimpleNet2):
  NORM_TYPE = 'IN'


class SimpleNetBN2(SimpleNet2):
  NORM_TYPE = 'BN'


class SimpleNetBN2B(SimpleNet2):
  NORM_TYPE = 'BN'
  CHANNELS = [None, 32, 64, 128, 256]
  TR_CHANNELS = [None, 64, 64, 64, 64]


class SimpleNetBN2C(SimpleNet2):
  NORM_TYPE = 'BN'
  CHANNELS = [None, 32, 64, 128, 256]
  TR_CHANNELS = [None, 32, 64, 64, 128]


class SimpleNetBN2D(SimpleNet2):
  NORM_TYPE = 'BN'
  CHANNELS = [None, 32, 64, 128, 256]
  TR_CHANNELS = [None, 32, 64, 64, 128]


class SimpleNetBN2E(SimpleNet2):
  NORM_TYPE = 'BN'
  CHANNELS = [None, 16, 32, 64, 128]
  TR_CHANNELS = [None, 16, 32, 32, 64]


class SimpleNetIN2E(SimpleNetBN2E):
  NORM_TYPE = 'IN'


class SimpleNet3(ME.MinkowskiNetwork):
  NORM_TYPE = None
  CHANNELS = [None, 32, 64, 128, 256, 512]
  TR_CHANNELS = [None, 32, 32, 64, 64, 128]

  # To use the model, must call initialize_coords before forward pass.
  # Once data is processed, call clear to reset the model before calling initialize_coords
  def __init__(self, in_channels=3, out_channels=32, bn_momentum=0.1, D=3, config=None):
    ME.MinkowskiNetwork.__init__(self, D)
    NORM_TYPE = self.NORM_TYPE
    bn_momentum = config.bn_momentum
    CHANNELS = self.CHANNELS
    TR_CHANNELS = self.TR_CHANNELS
    self.normalize_feature = config.normalize_feature
    self.conv1 = ME.MinkowskiConvolution(
        in_channels=in_channels,
        out_channels=CHANNELS[1],
        kernel_size=config.conv1_kernel_size,
        stride=1,
        dilation=1,
        has_bias=False,
        dimension=D)
    self.norm1 = get_norm(NORM_TYPE, CHANNELS[1], bn_momentum=bn_momentum, D=D)

    self.conv2 = ME.MinkowskiConvolution(
        in_channels=CHANNELS[1],
        out_channels=CHANNELS[2],
        kernel_size=3,
        stride=2,
        dilation=1,
        has_bias=False,
        dimension=D)
    self.norm2 = get_norm(NORM_TYPE, CHANNELS[2], bn_momentum=bn_momentum, D=D)

    self.conv3 = ME.MinkowskiConvolution(
        in_channels=CHANNELS[2],
        out_channels=CHANNELS[3],
        kernel_size=3,
        stride=2,
        dilation=1,
        has_bias=False,
        dimension=D)
    self.norm3 = get_norm(NORM_TYPE, CHANNELS[3], bn_momentum=bn_momentum, D=D)

    self.conv4 = ME.MinkowskiConvolution(
        in_channels=CHANNELS[3],
        out_channels=CHANNELS[4],
        kernel_size=3,
        stride=2,
        dilation=1,
        has_bias=False,
        dimension=D)
    self.norm4 = get_norm(NORM_TYPE, CHANNELS[4], bn_momentum=bn_momentum, D=D)

    self.conv5 = ME.MinkowskiConvolution(
        in_channels=CHANNELS[4],
        out_channels=CHANNELS[5],
        kernel_size=3,
        stride=2,
        dilation=1,
        has_bias=False,
        dimension=D)
    self.norm5 = get_norm(NORM_TYPE, CHANNELS[5], bn_momentum=bn_momentum, D=D)

    self.conv5_tr = ME.MinkowskiConvolutionTranspose(
        in_channels=CHANNELS[5],
        out_channels=TR_CHANNELS[5],
        kernel_size=3,
        stride=2,
        dilation=1,
        has_bias=False,
        dimension=D)
    self.norm5_tr = get_norm(NORM_TYPE, TR_CHANNELS[5], bn_momentum=bn_momentum, D=D)

    self.conv4_tr = ME.MinkowskiConvolutionTranspose(
        in_channels=CHANNELS[4] + TR_CHANNELS[5],
        out_channels=TR_CHANNELS[4],
        kernel_size=3,
        stride=2,
        dilation=1,
        has_bias=False,
        dimension=D)
    self.norm4_tr = get_norm(NORM_TYPE, TR_CHANNELS[4], bn_momentum=bn_momentum, D=D)

    self.conv3_tr = ME.MinkowskiConvolutionTranspose(
        in_channels=CHANNELS[3] + TR_CHANNELS[4],
        out_channels=TR_CHANNELS[3],
        kernel_size=3,
        stride=2,
        dilation=1,
        has_bias=False,
        dimension=D)
    self.norm3_tr = get_norm(NORM_TYPE, TR_CHANNELS[3], bn_momentum=bn_momentum, D=D)

    self.conv2_tr = ME.MinkowskiConvolutionTranspose(
        in_channels=CHANNELS[2] + TR_CHANNELS[3],
        out_channels=TR_CHANNELS[2],
        kernel_size=3,
        stride=2,
        dilation=1,
        has_bias=False,
        dimension=D)
    self.norm2_tr = get_norm(NORM_TYPE, TR_CHANNELS[2], bn_momentum=bn_momentum, D=D)

    self.conv1_tr = ME.MinkowskiConvolution(
        in_channels=CHANNELS[1] + TR_CHANNELS[2],
        out_channels=TR_CHANNELS[1],
        kernel_size=1,
        stride=1,
        dilation=1,
        has_bias=True,
        dimension=D)

  def forward(self, x):
    out_s1 = self.conv1(x)
    out_s1 = self.norm1(out_s1)
    out = MEF.relu(out_s1)

    out_s2 = self.conv2(out)
    out_s2 = self.norm2(out_s2)
    out = MEF.relu(out_s2)

    out_s4 = self.conv3(out)
    out_s4 = self.norm3(out_s4)
    out = MEF.relu(out_s4)

    out_s8 = self.conv4(out)
    out_s8 = self.norm4(out_s8)
    out = MEF.relu(out_s8)

    out_s16 = self.conv5(out)
    out_s16 = self.norm5(out_s16)
    out = MEF.relu(out_s16)

    out = self.conv5_tr(out)
    out = self.norm5_tr(out)
    out_s8_tr = MEF.relu(out)

    out = ME.cat((out_s8_tr, out_s8))

    out = self.conv4_tr(out)
    out = self.norm4_tr(out)
    out_s4_tr = MEF.relu(out)

    out = ME.cat((out_s4_tr, out_s4))

    out = self.conv3_tr(out)
    out = self.norm3_tr(out)
    out_s2_tr = MEF.relu(out)

    out = ME.cat((out_s2_tr, out_s2))

    out = self.conv2_tr(out)
    out = self.norm2_tr(out)
    out_s1_tr = MEF.relu(out)

    out = ME.cat((out_s1_tr, out_s1))
    out = self.conv1_tr(out)

    if self.normalize_feature:
      return ME.SparseTensor(
          out.F / torch.norm(out.F, p=2, dim=1, keepdim=True),
          coords_key=out.coords_key,
          coords_manager=out.coords_man)
    else:
      return out


class SimpleNetIN3(SimpleNet3):
  NORM_TYPE = 'IN'


class SimpleNetBN3(SimpleNet3):
  NORM_TYPE = 'BN'


class SimpleNetBN3B(SimpleNet3):
  NORM_TYPE = 'BN'
  CHANNELS = [None, 32, 64, 128, 256, 512]
  TR_CHANNELS = [None, 32, 64, 64, 64, 128]


class SimpleNetBN3C(SimpleNet3):
  NORM_TYPE = 'BN'
  CHANNELS = [None, 32, 64, 128, 256, 512]
  TR_CHANNELS = [None, 32, 32, 64, 128, 128]


class SimpleNetBN3D(SimpleNet3):
  NORM_TYPE = 'BN'
  CHANNELS = [None, 32, 64, 128, 256, 512]
  TR_CHANNELS = [None, 32, 64, 64, 128, 128]


class SimpleNetBN3E(SimpleNet3):
  NORM_TYPE = 'BN'
  CHANNELS = [None, 16, 32, 64, 128, 256]
  TR_CHANNELS = [None, 16, 32, 32, 64, 128]


class SimpleNetIN3E(SimpleNetBN3E):
  NORM_TYPE = 'IN'
