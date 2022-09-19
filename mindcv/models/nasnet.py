from typing import Optional, Type, List, Union, Dict, cast, Any

import mindspore.nn as nn
from mindspore import Tensor

from .utils import load_pretrained
from .registry import register_model

import numpy as np
import mindspore.ops.operations as P

__all__ = ['nasnet']

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000,
        'first_conv': '', 'classifier': '',
        **kwargs
    }


default_cfgs = {
    'nasnet': _cfg(url='')
}

class AuxLogits(nn.Cell):

    def __init__(self, in_channels, out_channels, name=None):
        super(AuxLogits, self).__init__()
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(5, stride=3, pad_mode='valid')
        self.conv = nn.Conv2d(in_channels, 128, kernel_size=1)
        self.bn = nn.BatchNorm2d(128)
        self.conv_1 = nn.Conv2d(128, 768, (4, 4), pad_mode='valid')
        self.bn_1 = nn.BatchNorm2d(768)
        self.flatten = nn.Flatten()
        if name == 'large':
            self.fc = nn.Dense(6912, out_channels)  # large: 6912, mobile:768
        else:
            self.fc = nn.Dense(768, out_channels)

    def construct(self, x):
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class SeparableConv2d(nn.Cell):

    def __init__(self, in_channels, out_channels, dw_kernel, dw_stride, dw_padding, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise_conv2d = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=dw_kernel,
                                          stride=dw_stride, pad_mode='pad', padding=dw_padding, group=in_channels,
                                          has_bias=bias)
        self.pointwise_conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1,
                                          pad_mode='pad', has_bias=bias)

    def construct(self, x):
        x = self.depthwise_conv2d(x)
        x = self.pointwise_conv2d(x)
        return x


class BranchSeparables(nn.Cell):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super(BranchSeparables, self).__init__()
        self.relu = nn.ReLU()
        self.separable_1 = SeparableConv2d(
            in_channels, in_channels, kernel_size, stride, padding, bias=bias
        )
        self.bn_sep_1 = nn.BatchNorm2d(num_features=in_channels, eps=0.001, momentum=0.9, affine=True)
        self.relu1 = nn.ReLU()
        self.separable_2 = SeparableConv2d(
            in_channels, out_channels, kernel_size, 1, padding, bias=bias
        )
        self.bn_sep_2 = nn.BatchNorm2d(num_features=out_channels, eps=0.001, momentum=0.9, affine=True)

    def construct(self, x):
        x = self.relu(x)
        x = self.separable_1(x)
        x = self.bn_sep_1(x)
        x = self.relu1(x)
        x = self.separable_2(x)
        x = self.bn_sep_2(x)
        return x


class BranchSeparablesStem(nn.Cell):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super(BranchSeparablesStem, self).__init__()
        self.relu = nn.ReLU()
        self.separable_1 = SeparableConv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=bias
        )
        self.bn_sep_1 = nn.BatchNorm2d(num_features=out_channels, eps=0.001, momentum=0.9, affine=True)
        self.relu1 = nn.ReLU()
        self.separable_2 = SeparableConv2d(
            out_channels, out_channels, kernel_size, 1, padding, bias=bias
        )
        self.bn_sep_2 = nn.BatchNorm2d(num_features=out_channels, eps=0.001, momentum=0.9, affine=True)

    def construct(self, x):
        x = self.relu(x)
        x = self.separable_1(x)
        x = self.bn_sep_1(x)
        x = self.relu1(x)
        x = self.separable_2(x)
        x = self.bn_sep_2(x)
        return x


class BranchSeparablesReduction(BranchSeparables):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, z_padding=1, bias=False):
        BranchSeparables.__init__(
            self, in_channels, out_channels, kernel_size, stride, padding, bias
        )
        self.padding = nn.Pad(paddings=((0, 0), (0, 0), (z_padding, 0), (z_padding, 0)), mode="CONSTANT")

    def construct(self, x):
        x = self.relu(x)
        x = self.padding(x)
        x = self.separable_1(x)
        x = x[:, :, 1:, 1:]
        x = self.bn_sep_1(x)
        x = self.relu1(x)
        x = self.separable_2(x)
        x = self.bn_sep_2(x)
        return x


class CellStem0(nn.Cell):

    def __init__(self, stem_filters, num_filters=42):
        super(CellStem0, self).__init__()
        self.num_filters = num_filters
        self.stem_filters = stem_filters
        self.conv_1x1 = nn.SequentialCell([
            nn.ReLU(),
            nn.Conv2d(in_channels=self.stem_filters, out_channels=self.num_filters, kernel_size=1, stride=1,
                      pad_mode='pad', has_bias=False),
            nn.BatchNorm2d(num_features=self.num_filters, eps=0.001, momentum=0.9, affine=True)
        ])

        self.comb_iter_0_left = BranchSeparables(
            self.num_filters, self.num_filters, 5, 2, 2
        )
        self.comb_iter_0_right = BranchSeparablesStem(
            self.stem_filters, self.num_filters, 7, 2, 3, bias=False
        )

        self.comb_iter_1_left = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')
        self.comb_iter_1_right = BranchSeparablesStem(
            self.stem_filters, self.num_filters, 7, 2, 3, bias=False
        )

        self.comb_iter_2_left = nn.AvgPool2d(kernel_size=3, stride=2, pad_mode='same')
        self.comb_iter_2_right = BranchSeparablesStem(
            self.stem_filters, self.num_filters, 5, 2, 2, bias=False
        )

        self.comb_iter_3_right = nn.AvgPool2d(kernel_size=3, stride=1, pad_mode='same')

        self.comb_iter_4_left = BranchSeparables(
            self.num_filters, self.num_filters, 3, 1, 1, bias=False
        )
        self.comb_iter_4_right = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')

    def construct(self, x):
        x1 = self.conv_1x1(x)

        x_comb_iter_0_left = self.comb_iter_0_left(x1)
        x_comb_iter_0_right = self.comb_iter_0_right(x)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right

        x_comb_iter_1_left = self.comb_iter_1_left(x1)
        x_comb_iter_1_right = self.comb_iter_1_right(x)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right

        x_comb_iter_2_left = self.comb_iter_2_left(x1)
        x_comb_iter_2_right = self.comb_iter_2_right(x)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right

        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1

        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x1)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right

        x_out = P.Concat(1)((x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4))
        return x_out


class CellStem1(nn.Cell):

    def __init__(self, stem_filters, num_filters):
        super(CellStem1, self).__init__()
        self.num_filters = num_filters
        self.stem_filters = stem_filters
        self.conv_1x1 = nn.SequentialCell([
            nn.ReLU(),
            nn.Conv2d(in_channels=2*self.num_filters, out_channels=self.num_filters, kernel_size=1, stride=1,
                      pad_mode='pad', has_bias=False),
            nn.BatchNorm2d(num_features=self.num_filters, eps=0.001, momentum=0.9, affine=True)])

        self.relu = nn.ReLU()
        self.path_1 = nn.SequentialCell([
            nn.AvgPool2d(kernel_size=1, stride=2, pad_mode='valid'),
            nn.Conv2d(in_channels=self.stem_filters, out_channels=self.num_filters//2, kernel_size=1, stride=1,
                      pad_mode='pad', has_bias=False)])

        self.path_2 = nn.CellList([])
        self.path_2.append(nn.Pad(paddings=((0, 0), (0, 0), (0, 1), (0, 1)), mode="CONSTANT"))
        self.path_2.append(
            nn.AvgPool2d(kernel_size=1, stride=2, pad_mode='valid')
        )
        self.path_2.append(
            nn.Conv2d(in_channels=self.stem_filters, out_channels=self.num_filters//2, kernel_size=1, stride=1,
                      pad_mode='pad', has_bias=False)
        )

        self.final_path_bn = nn.BatchNorm2d(num_features=self.num_filters, eps=0.001, momentum=0.9, affine=True)

        self.comb_iter_0_left = BranchSeparables(
            self.num_filters,
            self.num_filters,
            5,
            2,
            2,
            bias=False
        )
        self.comb_iter_0_right = BranchSeparables(
            self.num_filters,
            self.num_filters,
            7,
            2,
            3,
            bias=False
        )

        self.comb_iter_1_left = nn.MaxPool2d(3, stride=2, pad_mode='same')
        self.comb_iter_1_right = BranchSeparables(
            self.num_filters,
            self.num_filters,
            7,
            2,
            3,
            bias=False
        )

        self.comb_iter_2_left = nn.AvgPool2d(3, stride=2, pad_mode='same')
        self.comb_iter_2_right = BranchSeparables(
            self.num_filters,
            self.num_filters,
            5,
            2,
            2,
            bias=False
        )

        self.comb_iter_3_right = nn.AvgPool2d(kernel_size=3, stride=1, pad_mode='same')

        self.comb_iter_4_left = BranchSeparables(
            self.num_filters,
            self.num_filters,
            3,
            1,
            1,
            bias=False
        )
        self.comb_iter_4_right = nn.MaxPool2d(3, stride=2, pad_mode='same')
        self.shape = P.Shape()

    def construct(self, x_conv0, x_stem_0):
        x_left = self.conv_1x1(x_stem_0)
        x_relu = self.relu(x_conv0)
        # path 1
        x_path1 = self.path_1(x_relu)
        # path 2
        x_path2 = self.path_2[0](x_relu)
        x_path2 = x_path2[:, :, 1:, 1:]
        x_path2 = self.path_2[1](x_path2)
        x_path2 = self.path_2[2](x_path2)
        # final path
        x_right = self.final_path_bn(P.Concat(1)((x_path1, x_path2)))

        x_comb_iter_0_left = self.comb_iter_0_left(x_left)
        x_comb_iter_0_right = self.comb_iter_0_right(x_right)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right

        x_comb_iter_1_left = self.comb_iter_1_left(x_left)
        x_comb_iter_1_right = self.comb_iter_1_right(x_right)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right

        x_comb_iter_2_left = self.comb_iter_2_left(x_left)
        x_comb_iter_2_right = self.comb_iter_2_right(x_right)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right

        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1

        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x_left)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right

        x_out = P.Concat(1)((x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4))
        return x_out


class FirstCell(nn.Cell):

    def __init__(self, in_channels_left, out_channels_left, in_channels_right, out_channels_right):
        super(FirstCell, self).__init__()
        self.conv_1x1 = nn.SequentialCell([
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels_right, out_channels=out_channels_right, kernel_size=1, stride=1,
                      pad_mode='pad', has_bias=False),
            nn.BatchNorm2d(num_features=out_channels_right, eps=0.001, momentum=0.9, affine=True)])

        self.relu = nn.ReLU()
        self.path_1 = nn.SequentialCell([
            nn.AvgPool2d(kernel_size=1, stride=2, pad_mode='valid'),
            nn.Conv2d(in_channels=in_channels_left, out_channels=out_channels_left, kernel_size=1, stride=1,
                      pad_mode='pad', has_bias=False)])

        self.path_2 = nn.CellList([])
        self.path_2.append(nn.Pad(paddings=((0, 0), (0, 0), (0, 1), (0, 1)), mode="CONSTANT"))
        self.path_2.append(
            nn.AvgPool2d(kernel_size=1, stride=2, pad_mode='valid')
        )
        self.path_2.append(
            nn.Conv2d(in_channels=in_channels_left, out_channels=out_channels_left, kernel_size=1, stride=1,
                      pad_mode='pad', has_bias=False)
        )

        self.final_path_bn = nn.BatchNorm2d(num_features=out_channels_left*2, eps=0.001, momentum=0.9, affine=True)

        self.comb_iter_0_left = BranchSeparables(
            out_channels_right, out_channels_right, 5, 1, 2, bias=False
        )
        self.comb_iter_0_right = BranchSeparables(
            out_channels_right, out_channels_right, 3, 1, 1, bias=False
        )

        self.comb_iter_1_left = BranchSeparables(
            out_channels_right, out_channels_right, 5, 1, 2, bias=False
        )
        self.comb_iter_1_right = BranchSeparables(
            out_channels_right, out_channels_right, 3, 1, 1, bias=False
        )

        self.comb_iter_2_left = nn.AvgPool2d(kernel_size=3, stride=1, pad_mode='same')

        self.comb_iter_3_left = nn.AvgPool2d(kernel_size=3, stride=1, pad_mode='same')
        self.comb_iter_3_right = nn.AvgPool2d(kernel_size=3, stride=1, pad_mode='same')

        self.comb_iter_4_left = BranchSeparables(
            out_channels_right, out_channels_right, 3, 1, 1, bias=False
        )

    def construct(self, x, x_prev):
        x_relu = self.relu(x_prev)
        x_path1 = self.path_1(x_relu)
        x_path2 = self.path_2[0](x_relu)
        x_path2 = x_path2[:, :, 1:, 1:]
        x_path2 = self.path_2[1](x_path2)
        x_path2 = self.path_2[2](x_path2)
        # final path
        x_left = self.final_path_bn(P.Concat(1)((x_path1, x_path2)))

        x_right = self.conv_1x1(x)

        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right

        x_comb_iter_1_left = self.comb_iter_1_left(x_left)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right

        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2 = x_comb_iter_2_left + x_left

        x_comb_iter_3_left = self.comb_iter_3_left(x_left)
        x_comb_iter_3_right = self.comb_iter_3_right(x_left)
        x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right

        x_comb_iter_4_left = self.comb_iter_4_left(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_right

        x_out = P.Concat(1)((x_left, x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4))
        return x_out


class NormalCell(nn.Cell):

    def __init__(self, in_channels_left, out_channels_left, in_channels_right, out_channels_right):
        super(NormalCell, self).__init__()
        self.conv_prev_1x1 = nn.SequentialCell([
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels_left, out_channels=out_channels_left, kernel_size=1, stride=1,
                      pad_mode='pad', has_bias=False),
            nn.BatchNorm2d(num_features=out_channels_left, eps=0.001, momentum=0.9, affine=True)])

        self.conv_1x1 = nn.SequentialCell([
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels_right, out_channels=out_channels_right, kernel_size=1, stride=1,
                      pad_mode='pad', has_bias=False),
            nn.BatchNorm2d(num_features=out_channels_right, eps=0.001, momentum=0.9, affine=True)])

        self.comb_iter_0_left = BranchSeparables(
            out_channels_right, out_channels_right, 5, 1, 2, bias=False
        )
        self.comb_iter_0_right = BranchSeparables(
            out_channels_left, out_channels_left, 3, 1, 1, bias=False
        )

        self.comb_iter_1_left = BranchSeparables(
            out_channels_left, out_channels_left, 5, 1, 2, bias=False
        )
        self.comb_iter_1_right = BranchSeparables(
            out_channels_left, out_channels_left, 3, 1, 1, bias=False
        )

        self.comb_iter_2_left = nn.AvgPool2d(kernel_size=3, stride=1, pad_mode='same')

        self.comb_iter_3_left = nn.AvgPool2d(kernel_size=3, stride=1, pad_mode='same')
        self.comb_iter_3_right = nn.AvgPool2d(kernel_size=3, stride=1, pad_mode='same')

        self.comb_iter_4_left = BranchSeparables(
            out_channels_right, out_channels_right, 3, 1, 1, bias=False
        )

    def construct(self, x, x_prev):
        x_left = self.conv_prev_1x1(x_prev)
        x_right = self.conv_1x1(x)

        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right

        x_comb_iter_1_left = self.comb_iter_1_left(x_left)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right

        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2 = x_comb_iter_2_left + x_left

        x_comb_iter_3_left = self.comb_iter_3_left(x_left)
        x_comb_iter_3_right = self.comb_iter_3_right(x_left)
        x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right

        x_comb_iter_4_left = self.comb_iter_4_left(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_right

        x_out = P.Concat(1)((x_left, x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4))
        return x_out


class ReductionCell0(nn.Cell):

    def __init__(self, in_channels_left, out_channels_left, in_channels_right, out_channels_right):
        super(ReductionCell0, self).__init__()
        self.conv_prev_1x1 = nn.SequentialCell([
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels_left, out_channels=out_channels_left, kernel_size=1, stride=1,
                      pad_mode='pad', has_bias=False),
            nn.BatchNorm2d(num_features=out_channels_left, eps=0.001, momentum=0.9, affine=True)])

        self.conv_1x1 = nn.SequentialCell([
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels_right, out_channels=out_channels_right, kernel_size=1, stride=1,
                      pad_mode='pad', has_bias=False),
            nn.BatchNorm2d(num_features=out_channels_right, eps=0.001, momentum=0.9, affine=True)])

        self.comb_iter_0_left = BranchSeparablesReduction(
            out_channels_right, out_channels_right, 5, 2, 2, bias=False
        )
        self.comb_iter_0_right = BranchSeparablesReduction(
            out_channels_right, out_channels_right, 7, 2, 3, bias=False
        )

        self.comb_iter_1_left = nn.MaxPool2d(3, stride=2, pad_mode='same')
        self.comb_iter_1_right = BranchSeparablesReduction(
            out_channels_right, out_channels_right, 7, 2, 3, bias=False
        )

        self.comb_iter_2_left = nn.AvgPool2d(3, stride=2, pad_mode='same')
        self.comb_iter_2_right = BranchSeparablesReduction(
            out_channels_right, out_channels_right, 5, 2, 2, bias=False
        )

        self.comb_iter_3_right = nn.AvgPool2d(kernel_size=3, stride=1, pad_mode='same')

        self.comb_iter_4_left = BranchSeparablesReduction(
            out_channels_right, out_channels_right, 3, 1, 1, bias=False
        )
        self.comb_iter_4_right = nn.MaxPool2d(3, stride=2, pad_mode='same')

    def construct(self, x, x_prev):
        x_left = self.conv_prev_1x1(x_prev)
        x_right = self.conv_1x1(x)

        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right

        x_comb_iter_1_left = self.comb_iter_1_left(x_right)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right

        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2_right = self.comb_iter_2_right(x_left)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right

        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1

        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right

        x_out = P.Concat(1)((x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4))
        return x_out


class ReductionCell1(nn.Cell):

    def __init__(self, in_channels_left, out_channels_left, in_channels_right, out_channels_right):
        super(ReductionCell1, self).__init__()
        self.conv_prev_1x1 = nn.SequentialCell([
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels_left, out_channels=out_channels_left, kernel_size=1, stride=1,
                      pad_mode='pad', has_bias=False),
            nn.BatchNorm2d(num_features=out_channels_left, eps=0.001, momentum=0.9, affine=True)])

        self.conv_1x1 = nn.SequentialCell([
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels_right, out_channels=out_channels_right, kernel_size=1, stride=1,
                      pad_mode='pad', has_bias=False),
            nn.BatchNorm2d(num_features=out_channels_right, eps=0.001, momentum=0.9, affine=True)])

        self.comb_iter_0_left = BranchSeparables(
            out_channels_right,
            out_channels_right,
            5,
            2,
            2,
            bias=False
        )
        self.comb_iter_0_right = BranchSeparables(
            out_channels_right,
            out_channels_right,
            7,
            2,
            3,
            bias=False
        )

        self.comb_iter_1_left = nn.MaxPool2d(3, stride=2, pad_mode='same')
        self.comb_iter_1_right = BranchSeparables(
            out_channels_right,
            out_channels_right,
            7,
            2,
            3,
            bias=False
        )

        self.comb_iter_2_left = nn.AvgPool2d(3, stride=2, pad_mode='same')
        self.comb_iter_2_right = BranchSeparables(
            out_channels_right,
            out_channels_right,
            5,
            2,
            2,
            bias=False
        )

        self.comb_iter_3_right = nn.AvgPool2d(kernel_size=3, stride=1, pad_mode='same')

        self.comb_iter_4_left = BranchSeparables(
            out_channels_right,
            out_channels_right,
            3,
            1,
            1,
            bias=False
        )
        self.comb_iter_4_right = nn.MaxPool2d(3, stride=2, pad_mode='same')

    def construct(self, x, x_prev):
        x_left = self.conv_prev_1x1(x_prev)
        x_right = self.conv_1x1(x)

        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right

        x_comb_iter_1_left = self.comb_iter_1_left(x_right)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right

        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2_right = self.comb_iter_2_right(x_left)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right

        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1

        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right

        x_out = P.Concat(1)((x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4))
        return x_out

class NASNetAMobile(nn.Cell):
    """Neural Architecture Search (NAS).

    Reference:
        Zoph et al. Learning Transferable Architectures
        for Scalable Image Recognition. CVPR 2018.
        - ``nasnetamobile``: NASNet-A Mobile.
    """

    def __init__(self, num_classes, is_training=True,
                 stem_filters=32, penultimate_filters=1056, filters_multiplier=2):
        super(NASNetAMobile, self).__init__()
        self.is_training = is_training
        self.stem_filters = stem_filters
        self.penultimate_filters = penultimate_filters
        self.filters_multiplier = filters_multiplier

        filters = self.penultimate_filters//24
        # 24 is default value for the architecture

        self.conv0 = nn.SequentialCell([
            nn.Conv2d(in_channels=3, out_channels=self.stem_filters, kernel_size=3, stride=2, pad_mode='pad', padding=0,
                      has_bias=False),
            nn.BatchNorm2d(num_features=self.stem_filters, eps=0.001, momentum=0.9, affine=True)
        ])

        self.cell_stem_0 = CellStem0(
            self.stem_filters, num_filters=filters//(filters_multiplier**2)
        )
        self.cell_stem_1 = CellStem1(
            self.stem_filters, num_filters=filters//filters_multiplier
        )

        self.cell_0 = FirstCell(
            in_channels_left=filters,
            out_channels_left=filters//2,  # 1, 0.5
            in_channels_right=2*filters,
            out_channels_right=filters
        )  # 2, 1
        self.cell_1 = NormalCell(
            in_channels_left=2*filters,
            out_channels_left=filters,  # 2, 1
            in_channels_right=6*filters,
            out_channels_right=filters
        )  # 6, 1
        self.cell_2 = NormalCell(
            in_channels_left=6*filters,
            out_channels_left=filters,  # 6, 1
            in_channels_right=6*filters,
            out_channels_right=filters
        )  # 6, 1
        self.cell_3 = NormalCell(
            in_channels_left=6*filters,
            out_channels_left=filters,  # 6, 1
            in_channels_right=6*filters,
            out_channels_right=filters
        )  # 6, 1

        self.reduction_cell_0 = ReductionCell0(
            in_channels_left=6*filters,
            out_channels_left=2*filters,  # 6, 2
            in_channels_right=6*filters,
            out_channels_right=2*filters
        )  # 6, 2

        self.cell_6 = FirstCell(
            in_channels_left=6*filters,
            out_channels_left=filters,  # 6, 1
            in_channels_right=8*filters,
            out_channels_right=2*filters
        )  # 8, 2
        self.cell_7 = NormalCell(
            in_channels_left=8*filters,
            out_channels_left=2*filters,  # 8, 2
            in_channels_right=12*filters,
            out_channels_right=2*filters
        )  # 12, 2
        self.cell_8 = NormalCell(
            in_channels_left=12*filters,
            out_channels_left=2*filters,  # 12, 2
            in_channels_right=12*filters,
            out_channels_right=2*filters
        )  # 12, 2
        self.cell_9 = NormalCell(
            in_channels_left=12*filters,
            out_channels_left=2*filters,  # 12, 2
            in_channels_right=12*filters,
            out_channels_right=2*filters
        )  # 12, 2

        if is_training:
            self.aux_logits = AuxLogits(in_channels=12*filters, out_channels=num_classes)

        self.reduction_cell_1 = ReductionCell1(
            in_channels_left=12*filters,
            out_channels_left=4*filters,  # 12, 4
            in_channels_right=12*filters,
            out_channels_right=4*filters
        )  # 12, 4

        self.cell_12 = FirstCell(
            in_channels_left=12*filters,
            out_channels_left=2*filters,  # 12, 2
            in_channels_right=16*filters,
            out_channels_right=4*filters
        )  # 16, 4
        self.cell_13 = NormalCell(
            in_channels_left=16*filters,
            out_channels_left=4*filters,  # 16, 4
            in_channels_right=24*filters,
            out_channels_right=4*filters
        )  # 24, 4
        self.cell_14 = NormalCell(
            in_channels_left=24*filters,
            out_channels_left=4*filters,  # 24, 4
            in_channels_right=24*filters,
            out_channels_right=4*filters
        )  # 24, 4
        self.cell_15 = NormalCell(
            in_channels_left=24*filters,
            out_channels_left=4*filters,  # 24, 4
            in_channels_right=24*filters,
            out_channels_right=4*filters
        )  # 24, 4

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(keep_prob=0.5)
        self.classifier = nn.Dense(in_channels=24*filters, out_channels=num_classes)
        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=1)
        self._initialize_weights()

    def _initialize_weights(self):
        self.init_parameters_data()
        for _, m in self.cells_and_names():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0]*m.kernel_size[1]*m.out_channels
                m.weight.set_data(Tensor(np.random.normal(0, np.sqrt(2./n),
                                                          m.weight.data.shape).astype("float32")))
                if m.bias is not None:
                    m.bias.set_data(
                        Tensor(np.zeros(m.bias.data.shape, dtype="float32")))
            elif isinstance(m, nn.BatchNorm2d):
                m.gamma.set_data(
                    Tensor(np.ones(m.gamma.data.shape, dtype="float32")))
                m.beta.set_data(
                    Tensor(np.zeros(m.beta.data.shape, dtype="float32")))
            elif isinstance(m, nn.Dense):
                m.weight.set_data(Tensor(np.random.normal(
                    0, 0.01, m.weight.data.shape).astype("float32")))
                if m.bias is not None:
                    m.bias.set_data(
                        Tensor(np.zeros(m.bias.data.shape, dtype="float32")))

    def forward_features(self, x: Tensor) -> Tensor:
        x_conv0 = self.conv0(x)
        x_stem_0 = self.cell_stem_0(x_conv0)
        x_stem_1 = self.cell_stem_1(x_conv0, x_stem_0)

        x_cell_0 = self.cell_0(x_stem_1, x_stem_0)
        x_cell_1 = self.cell_1(x_cell_0, x_stem_1)
        x_cell_2 = self.cell_2(x_cell_1, x_cell_0)
        x_cell_3 = self.cell_3(x_cell_2, x_cell_1)

        x_reduction_cell_0 = self.reduction_cell_0(x_cell_3, x_cell_2)

        x_cell_6 = self.cell_6(x_reduction_cell_0, x_cell_3)
        x_cell_7 = self.cell_7(x_cell_6, x_reduction_cell_0)
        x_cell_8 = self.cell_8(x_cell_7, x_cell_6)
        x_cell_9 = self.cell_9(x_cell_8, x_cell_7)

        x_reduction_cell_1 = self.reduction_cell_1(x_cell_9, x_cell_8)

        x_cell_12 = self.cell_12(x_reduction_cell_1, x_cell_9)
        x_cell_13 = self.cell_13(x_cell_12, x_reduction_cell_1)
        x_cell_14 = self.cell_14(x_cell_13, x_cell_12)
        x_cell_15 = self.cell_15(x_cell_14, x_cell_13)

        x_cell_15 = self.relu(x_cell_15)
        return x_cell_15

    def forward_head(self, x: Tensor) -> Tensor:
        x = self.avg_pool(x)  # global average pool
        x = self.reshape(x, (self.shape(x)[0], -1,))
        x = self.dropout(x)
        x = self.classifier(x)
        return x

    def construct(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x

@register_model
def nasnet(pretrained: bool = False, num_classes: int = 1000, in_channels=3) -> NASNetAMobile:
    """NASNet-A large model architecture.
    """
    default_cfg = default_cfgs['nasnet']
    model = NASNetAMobile(num_classes=1000, is_training=False)
    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model
