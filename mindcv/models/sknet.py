from typing import Optional, Dict

import mindspore.nn as nn
from mindspore import Tensor

from .layers.conv_norm_act import Conv2dNormActivation
from .layers.selective_kernel import SelectiveKernel
from .utils import load_pretrained
from .registry import register_model
from .resnet import ResNet

__all__ = [
]


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000,
        'first_conv': '', 'classifier': '',
        **kwargs
    }


default_cfgs = {
    'sk_resnet18': _cfg(url=''),
    'sk_resnet34': _cfg(url=''),
    'sk_resnet50': _cfg(url=''),
    'sk_resnext50_32X4d': _cfg(url='')

}


class SelectiveKernelBasic(nn.Cell):
    expansion = 1

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 groups: int = 1,
                 down_sample: Optional[nn.Cell] = None,
                 base_width: int = 64,
                 norm: Optional[nn.Cell] = None,
                 block_kwargs: Optional[Dict] = None
                 ):
        super(SelectiveKernelBasic, self).__init__()
        if norm is None:
            norm = nn.BatchNorm2d

        if block_kwargs is None:
            sk_kwargs = {}
        else:
            sk_kwargs = block_kwargs['sk_kwargs']

        assert groups == 1, 'BasicBlock only supports cardinality of 1'
        assert base_width == 64, 'BasicBlock doest not support changing base width'

        self.conv1 = SelectiveKernel(
            in_channels, out_channels, stride=stride, **sk_kwargs)
        self.conv2 = Conv2dNormActivation(
            out_channels, out_channels * self.expansion, kernel_size=3, padding=1, norm=norm)

        self.relu = nn.ReLU()
        self.down_sample = down_sample

    def construct(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.down_sample:
            identity = self.down_sample(x)
        out += identity
        out = self.relu(out)
        return out


class SelectiveKernelBottleneck(nn.Cell):
    expansion = 4

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 down_sample: Optional[nn.Cell] = None,
                 groups: int = 1,
                 base_width: int = 64,
                 norm: Optional[nn.Cell] = None,
                 block_kwargs: Optional[Dict] = None,

                 ):
        super(SelectiveKernelBottleneck, self).__init__()
        if norm is None:
            norm = nn.BatchNorm2d

        if block_kwargs is None:
            sk_kwargs = {}
        else:
            sk_kwargs = block_kwargs['sk_kwargs']

        width = int(out_channels * (base_width / 64.0)) * groups
        self.conv1 = Conv2dNormActivation(
            in_channels, width, kernel_size=1, padding=0, norm=norm)
        self.conv2 = SelectiveKernel(
            width, width, stride=stride, groups=groups, **sk_kwargs)
        self.conv3 = Conv2dNormActivation(
            width, out_channels * self.expansion, kernel_size=1, padding=0, norm=norm)
        self.relu = nn.ReLU()
        self.down_sample = down_sample

    def construct(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.down_sample:
            identity = self.down_sample(x)
        out += identity
        out = self.relu(out)
        return out


@register_model
def sk_resnet18(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> ResNet:
    default_cfg = default_cfgs['sk_resnet18']
    sk_kwargs = dict(rd_ratio=1 / 8, rd_divisor=16, split_input=True)
    model = ResNet(SelectiveKernelBasic, [2, 2, 2, 2], num_classes=num_classes, in_channels=in_channels,
                   block_kwargs=dict(sk_kwargs=sk_kwargs), **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def sk_resnet34(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> ResNet:
    default_cfg = default_cfgs['sk_resnet34']
    sk_kwargs = dict(rd_ratio=1 / 8, rd_divisor=16, split_input=True)
    model = ResNet(SelectiveKernelBasic, [3, 4, 6, 3], num_classes=num_classes, in_channels=in_channels,
                   block_kwargs=dict(sk_kwargs=sk_kwargs), **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def sk_resnet50(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> ResNet:
    default_cfg = default_cfgs['sk_resnet50']
    sk_kwargs = dict(split_input=True)
    model = ResNet(SelectiveKernelBottleneck, [3, 4, 6, 3], num_classes=num_classes, in_channels=in_channels,
                   block_kwargs=dict(sk_kwargs=sk_kwargs), **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def sk_resnext50_32x4d(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> ResNet:
    default_cfg = default_cfgs['sk_resnext50_32X4d']
    sk_kwargs = dict(rd_ratio=1 / 16, rd_divisor=32, split_input=False)
    model = ResNet(SelectiveKernelBottleneck, [3, 4, 6, 3], num_classes=num_classes, in_channels=in_channels,
                   block_kwargs=dict(sk_kwargs=sk_kwargs), **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model
