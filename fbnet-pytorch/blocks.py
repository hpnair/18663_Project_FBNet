import torch
import torch.nn as nn


class ChannelShuffle(nn.Module):
  def __init__(self, group=1):
    assert group > 1
    super(ChannelShuffle, self).__init__()
    self.group = group
  def forward(self, x):
    """https://github.com/Randl/ShuffleNetV2-pytorch/blob/master/model.py
    """
    batchsize, num_channels, height, width = x.data.size()
    assert (num_channels % self.group == 0)
    channels_per_group = num_channels // self.group
    # reshape
    x = x.view(batchsize, self.group, channels_per_group, height, width)
    # transpose
    # - contiguous() required if transpose() is used before view().
    #   See https://github.com/pytorch/pytorch/issues/764
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x

class Identity(nn.Module):
  def __init__(self):
    super(Identity, self).__init__()
  def forward(self, x):
    return x

class FBNetBlock(nn.Module):
  def __init__(self, C_in, C_out, kernel_size, stride,
              expansion, group, bn=False):
    super(FBNetBlock, self).__init__()
    assert not bn, "not support bn for now"
    bias_flag = not bn
    if kernel_size == 1:
      padding = 0
    elif kernel_size == 3:
      padding = 1
    elif kernel_size == 5:
      padding = 2
    elif kernel_size == 7:
      padding = 3
    else:
      raise ValueError("Not supported kernel_size %d" % kernel_size)
    if group == 1:
      self.op = nn.Sequential(
        nn.Conv2d(C_in, C_in*expansion, 1, stride=1, padding=0,
                  groups=group, bias=bias_flag),
        nn.ReLU(inplace=False),
        nn.Conv2d(C_in*expansion, C_in*expansion, kernel_size, stride=stride, 
                  padding=padding, groups=C_in*expansion, bias=bias_flag),
        nn.ReLU(inplace=False),
        nn.Conv2d(C_in*expansion, C_out, 1, stride=1, padding=0, 
                  groups=group, bias=bias_flag)
      )
    else:
      self.op = nn.Sequential(
        nn.Conv2d(C_in, C_in*expansion, 1, stride=1, padding=0,
                  groups=group, bias=bias_flag),
        nn.ReLU(inplace=False),
        ChannelShuffle(group),
        nn.Conv2d(C_in*expansion, C_in*expansion, kernel_size, stride=stride, 
                  padding=padding, groups=C_in*expansion, bias=bias_flag),
        nn.ReLU(inplace=False),
        nn.Conv2d(C_in*expansion, C_out, 1, stride=1, padding=0, 
                  groups=group, bias=bias_flag),
        ChannelShuffle(group)
      )
    res_flag = ((C_in == C_out) and (stride == 1))
    self.res_flag = res_flag
    # if not res_flag:
    #   if stride == 2:
    #     self.trans = nn.Conv2d(C_in, C_out, 3, stride=2, 
    #                           padding=1)
    #   elif stride == 1:
    #     self.trans = nn.Conv2d(C_in, C_out, 1, stride=1, 
    #                           padding=0)
    #   else:
    #     raise ValueError("Wrong stride %d provided" % stride)

  def forward(self, x):
    if self.res_flag:
      return self.op(x) + x
    else:
      return self.op(x) # + self.trans(x)

def get_blocks(cifar10=False, face=False):
  BLOCKS = []
  _f = [16, 16, 24, 32, 
      64, 112, 184, 352,
      1984]
  _n = [1, 1, 4, 4,
      4, 4, 4, 1,
      1]
  if cifar10:
    assert not face
    _s = [1, 1, 2, 2,
        1, 1, 1, 1,
        1]
  elif face:
    assert not cifar10
    _s = [1, 1, 2, 2,
        2, 1, 2, 1,
        1]
  else:
    _s = [2, 1, 2, 2,
        2, 1, 2, 1,
        1]
  _e = [1, 1, 3, 6,
      1, 1, 3, 6]
  _kernel = [3, 3, 3, 3,
          5, 5, 5, 5]
  _group = [1, 2, 1, 1,
          1, 2, 1, 1]
  tbs_range = slice(1, 8) # [1, 7]

  BLOCKS.append(nn.Conv2d(3, 16, 3, 2, padding=1))
  
  c_in = 16
  for n_idx in range(len(_n))[tbs_range]:
    c_out = _f[n_idx]
    stride = _s[n_idx]

    for inner_idx in range(_n[n_idx]):
      # c_out = _f[n_idx]
      tmp_block = []

      for b_idx in range(len(_e)):
        expansion = _e[b_idx]
        kernel = _kernel[b_idx]
        group = _group[b_idx]

        tmp_block.append(FBNetBlock(c_in, c_out,
                kernel, stride, expansion, group))
      if inner_idx > 0 and ((c_in == c_out) and (stride == 1)):
        tmp_block.append(Identity())

      BLOCKS.append(tmp_block)
      stride = 1
      c_in = c_out
  BLOCKS.append(nn.Conv2d(c_out, 1984, 1, padding=0))
  assert len(BLOCKS) == 24
  return BLOCKS
