from typing import Optional, Union, List

import mindspore.nn as nn
import mindspore.ops as ops

from .conv_norm_act import Conv2dNormActivation
from .pooling import GlobalAvgPooling
from ..utils import make_divisible


def _kernel_valid(k):
    if isinstance(k, (list, tuple)):
        for ki in k:
            return _kernel_valid(ki)
    assert k >= 3 and k % 2


class SelectiveKernelAttn(nn.Cell):
    def __init__(self,
                 channels: int,
                 num_paths: int = 2,
                 attn_channels: int = 32,
                 activation: Optional[nn.Cell] = nn.ReLU,
                 norm: Optional[nn.Cell] = nn.BatchNorm2d
                 ):
        super(SelectiveKernelAttn, self).__init__()
        self.num_paths = num_paths
        self.mean = GlobalAvgPooling(keep_dims=True)
        self.fc_reduce = nn.Conv2d(channels, attn_channels, kernel_size=1, has_bias=False)
        self.bn = norm(attn_channels)
        self.act = activation()
        self.fc_select = nn.Conv2d(attn_channels, channels * num_paths, kernel_size=1)
        self.softmax = ops.Softmax(axis=1)

    def construct(self, x):
        x = self.mean((x.sum(1)))
        x = self.fc_reduce(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.fc_select(x)
        B, C, H, W = x.shape
        x = x.reshape((B, self.num_paths, C // self.num_paths, H, W))
        x = self.softmax(x)
        return x


class SelectiveKernel(nn.Cell):

    def __init__(self,
                 in_channels: int,
                 out_channels: Optional[int] = None,
                 kernel_size: Optional[Union[int, List]] = None,
                 stride: int = 1,
                 dilation: int = 1,
                 groups: int = 1,
                 rd_ratio: float = 1. / 16,
                 rd_channels: Optional[int] = None,
                 rd_divisor: int = 8,
                 keep_3x3: bool = True,
                 split_input: bool = True,
                 activation: Optional[nn.Cell] = nn.ReLU,
                 norm: Optional[nn.Cell] = nn.BatchNorm2d
                 ):

        super(SelectiveKernel, self).__init__()
        out_channels = out_channels or in_channels
        kernel_size = kernel_size or [3, 5]  # default to one 3x3 and one 5x5 branch. 5x5 -> 3x3 + dilation
        _kernel_valid(kernel_size)
        if not isinstance(kernel_size, list):
            kernel_size = [kernel_size] * 2
        if keep_3x3:
            dilation = [dilation * (k - 1) // 2 for k in kernel_size]
            kernel_size = [3] * len(kernel_size)
        else:
            dilation = [dilation] * len(kernel_size)
        self.num_paths = len(kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.split_input = split_input
        if self.split_input:
            assert in_channels % self.num_paths == 0
            in_channels = in_channels // self.num_paths
        groups = min(out_channels, groups)

        self.paths = nn.CellList([
            Conv2dNormActivation(in_channels, out_channels, kernel_size=k, stride=stride, groups=groups,
                                 dilation=d, activation=activation, norm=norm)
            for k, d in zip(kernel_size, dilation)
        ])

        attn_channels = rd_channels or make_divisible(out_channels * rd_ratio, divisor=rd_divisor)
        self.attn = SelectiveKernelAttn(out_channels, self.num_paths, attn_channels)

    def construct(self, x):

        x_paths = []
        if self.split_input:
            x_split = ops.split(x, axis=1, output_num=self.num_paths)
            for i, op in enumerate(self.paths):
                x_paths.append(op(x_split[i]))
        else:
            for op in self.paths:
                x_paths.append(op(x))

        x = ops.stack(x_paths, axis=1)
        x_attn = self.attn(x)
        x = x * x_attn
        x = x.sum(1)
        return x
