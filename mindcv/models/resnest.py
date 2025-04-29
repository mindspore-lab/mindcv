"""
MindSpore implementation of `ResNeSt`.
Refer to ResNeSt: Split-Attention Networks.
"""

from typing import List, Optional, Type

import mindspore.common.initializer as init
from mindspore import Tensor, mint, nn

from .helpers import build_model_with_cfg, make_divisible
from .layers.identity import Identity
from .layers.pooling import GlobalAvgPooling
from .registry import register_model

__all__ = [
    "ResNeSt",
    "resnest14",
    "resnest26",
    "resnest50",
    "resnest101",
    "resnest200",
    "resnest269",
]


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "first_conv": "conv1.0",
        "classifier": "fc",
        **kwargs,
    }


default_cfgs = {
    "resnest14": _cfg(url=""),
    "resnest26": _cfg(url=""),
    "resnest50": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/resnest/resnest50-f2e7fc9c.ckpt"),
    "resnest101": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/resnest/resnest101-7cc5c258.ckpt"),
    "resnest200": _cfg(url=""),
    "resnest269": _cfg(url=""),
}


class RadixSoftmax(nn.Cell):
    def __init__(
            self,
            radix: int,
            cardinality: int
    ) -> None:
        super(RadixSoftmax, self).__init__()
        self.radix = radix
        self.cardinality = cardinality
        self.softmax = mint.nn.Softmax(dim=1)

    def construct(self, x: Tensor) -> Tensor:
        batch = x.shape[0]
        if self.radix > 1:
            x = mint.reshape(x, (batch, self.cardinality, self.radix, -1))
            x = mint.permute(x, (0, 2, 1, 3))
            x = self.softmax(x)
            x = mint.reshape(x, (batch, -1))
        else:
            x = mint.sigmoid(x)

        return x


class SplitAttn(nn.Cell):
    """Split-Attention Conv2d"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        group: int = 1,
        bias: bool = False,
        radix: int = 2,
        rd_ratio: float = 0.25,
        rd_channels: Optional[int] = None,
        rd_divisor: int = 8,
        act_layer: nn.Cell = mint.nn.ReLU,
        norm_layer: Optional[nn.Cell] = None,
    ) -> None:
        super(SplitAttn, self).__init__()
        out_channels = out_channels or in_channels
        self.radix = radix
        mid_chs = out_channels * radix

        if rd_channels is None:
            attn_chs = make_divisible(in_channels * radix * rd_ratio, min_value=32, divisor=rd_divisor)
        else:
            attn_chs = rd_channels * radix

        padding = kernel_size // 2 if padding is None else padding

        self.conv = mint.nn.Conv2d(
            in_channels, mid_chs, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=group * radix, bias=bias
        )
        self.bn0 = norm_layer(mid_chs) if norm_layer else Identity()
        self.act0 = act_layer()
        self.fc1 = mint.nn.Conv2d(out_channels, attn_chs, 1, groups=group, bias=True)
        self.bn1 = norm_layer(attn_chs) if norm_layer else nn.Identity()
        self.act1 = act_layer()
        self.fc2 = mint.nn.Conv2d(attn_chs, mid_chs, 1, groups=group, bias=True)
        self.rsoftmax = RadixSoftmax(radix, group)
        self.pool = GlobalAvgPooling(keep_dims=True)

    def construct(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn0(x)
        x = self.act0(x)

        B, RC, H, W = x.shape
        if self.radix > 1:
            x = mint.reshape(x, (B, self.radix, RC // self.radix, H, W))
            x_gap = mint.sum(x, dim=1)
        else:
            x_gap = x
        x_gap = self.pool(x_gap)
        x_gap = self.fc1(x_gap)
        x_gap = self.bn1(x_gap)
        x_gap = self.act1(x_gap)
        x_attn = self.fc2(x_gap)

        x_attn = self.rsoftmax(x_attn)
        x_attn = mint.reshape(x_attn, (B, -1, 1, 1))
        if self.radix > 1:
            out = x * mint.reshape(x_attn, (B, self.radix, RC // self.radix, 1, 1))
            out = mint.sum(out, dim=1)
        else:
            out = x * x_attn

        return out


class Bottleneck(nn.Cell):
    """ResNeSt Bottleneck"""

    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride=1,
        downsample: Optional[nn.SequentialCell] = None,
        radix: int = 1,
        cardinality: int = 1,
        bottleneck_width: int = 64,
        avd: bool = False,
        avd_first: bool = False,
        dilation: int = 1,
        is_first: bool = False,
        norm_layer: Optional[nn.Cell] = None,
    ) -> None:
        super(Bottleneck, self).__init__()
        group_width = int(planes * (bottleneck_width / 64.0)) * cardinality
        self.conv1 = mint.nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(group_width)
        self.radix = radix
        self.avd = avd and (stride > 1 or is_first)
        self.avd_first = avd_first

        if self.avd:
            self.avd_layer = mint.nn.AvgPool2d(3, stride, padding=1)
            stride = 1

        if radix >= 1:
            self.conv2 = SplitAttn(group_width, group_width, kernel_size=3, stride=stride,
                                   padding=dilation, dilation=dilation, group=cardinality,
                                   bias=False, radix=radix, norm_layer=norm_layer)
        else:
            self.conv2 = mint.nn.Conv2d(
                group_width, group_width, kernel_size=3, stride=stride, padding=dilation,
                dilation=dilation, groups=cardinality, bias=False
            )
            self.bn2 = norm_layer(group_width)

        self.conv3 = mint.nn.Conv2d(group_width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * 4)

        self.relu = mint.nn.ReLU()
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def construct(self, x: Tensor) -> Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.avd and self.avd_first:
            out = self.avd_layer(out)

        out = self.conv2(out)
        if self.radix == 0:
            out = self.bn2(out)
            out = self.relu(out)

        if self.avd and not self.avd_first:
            out = self.avd_layer(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNeSt(nn.Cell):
    r"""ResNeSt model class, based on
    `"ResNeSt: Split-Attention Networks" <https://arxiv.org/abs/2004.08955>`_

    Args:
        block: Class for the residual block. Option is Bottleneck.
        layers: Numbers of layers in each block.
        radix: Number of groups for Split-Attention conv. Default: 1.
        group: Number of groups for the conv in each bottleneck block. Default: 1.
        bottleneck_width: bottleneck channels factor. Default: 64.
        num_classes: Number of classification classes. Default: 1000.
        dilated: Applying dilation strategy to pretrained ResNeSt yielding a stride-8 model,
                 typically used in Semantic Segmentation. Default: False.
        dilation: Number of dilation in the conv. Default: 1.
        deep_stem: three 3x3 convolution layers of widths stem_width, stem_width, stem_width * 2.
                   Default: False.
        stem_width: number of channels in stem convolutions. Default: 64.
        avg_down: use avg pooling for projection skip connection between stages/downsample.
                  Default: False.
        avd: use avg pooling before or after split-attention conv. Default: False.
        avd_first: use avg pooling before or after split-attention conv. Default: False.
        drop_rate: Drop probability for the Dropout layer. Default: 0.
        norm_layer: Normalization layer used in backbone network. Default: nn.BatchNorm2d.
    """

    def __init__(
        self,
        block: Type[Bottleneck],
        layers: List[int],
        radix: int = 1,
        group: int = 1,
        bottleneck_width: int = 64,
        num_classes: int = 1000,
        dilated: bool = False,
        dilation: int = 1,
        deep_stem: bool = False,
        stem_width: int = 64,
        avg_down: bool = False,
        avd: bool = False,
        avd_first: bool = False,
        drop_rate: float = 0.0,
        norm_layer: nn.Cell = mint.nn.BatchNorm2d,
    ) -> None:
        super(ResNeSt, self).__init__()
        self.cardinality = group
        self.bottleneck_width = bottleneck_width
        # ResNet-D params
        self.inplanes = stem_width * 2 if deep_stem else 64
        self.avg_down = avg_down
        # ResNeSt params
        self.radix = radix
        self.avd = avd
        self.avd_first = avd_first

        if deep_stem:
            self.conv1 = nn.SequentialCell([
                mint.nn.Conv2d(3, stem_width, kernel_size=3, stride=2, padding=1, bias=False),
                norm_layer(stem_width),
                mint.nn.ReLU(),
                mint.nn.Conv2d(stem_width, stem_width, kernel_size=3, stride=1, padding=1, bias=False),
                norm_layer(stem_width),
                mint.nn.ReLU(),
                mint.nn.Conv2d(stem_width, stem_width * 2, kernel_size=3, stride=1, padding=1, bias=False),
            ])
        else:
            self.conv1 = mint.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.bn1 = norm_layer(self.inplanes)
        self.relu = mint.nn.ReLU()
        self.feature_info = [dict(chs=self.inplanes, reduction=2, name="relu")]
        self.maxpool = mint.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer, is_first=False)
        self.feature_info.append(dict(chs=block.expansion * 64, reduction=4, name='layer1'))
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        self.feature_info.append(dict(chs=block.expansion * 128, reduction=8, name='layer2'))

        if dilated or dilation == 4:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2, norm_layer=norm_layer)
            self.feature_info.append(dict(chs=block.expansion * 256, reduction=8, name='layer3'))
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4, norm_layer=norm_layer)
            self.feature_info.append(dict(chs=block.expansion * 512, reduction=8, name='layer4'))
        elif dilation == 2:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilation=1, norm_layer=norm_layer)
            self.feature_info.append(dict(chs=block.expansion * 256, reduction=16, name='layer3'))
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=2, norm_layer=norm_layer)
            self.feature_info.append(dict(chs=block.expansion * 512, reduction=16, name='layer4'))
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_layer=norm_layer)
            self.feature_info.append(dict(chs=block.expansion * 256, reduction=16, name='layer3'))
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2, norm_layer=norm_layer)
            self.feature_info.append(dict(chs=block.expansion * 512, reduction=32, name='layer4'))

        self.avgpool = GlobalAvgPooling()
        self.drop = mint.nn.Dropout(p=drop_rate) if drop_rate > 0.0 else None
        self.fc = mint.nn.Linear(512 * block.expansion, num_classes)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize weights for cells."""
        for _, cell in self.cells_and_names():
            if isinstance(cell, mint.nn.Conv2d):
                cell.weight.set_data(
                    init.initializer(
                        init.HeNormal(mode="fan_out", nonlinearity="relu"), cell.weight.shape, cell.weight.dtype
                    )
                )
                if cell.bias is not None:
                    cell.bias.set_data(init.initializer("zeros", cell.bias.shape, cell.bias.dtype))
            elif isinstance(cell, mint.nn.BatchNorm2d):
                cell.weight.set_data(init.initializer("ones", cell.weight.shape, cell.weight.dtype))
                cell.bias.set_data(init.initializer("zeros", cell.bias.shape, cell.bias.dtype))
            elif isinstance(cell, mint.nn.Linear):
                cell.weight.set_data(
                    init.initializer(
                        init.HeUniform(mode="fan_in", nonlinearity="sigmoid"), cell.weight.shape, cell.weight.dtype
                    )
                )
                if cell.bias is not None:
                    cell.bias.set_data(init.initializer("zeros", cell.bias.shape, cell.bias.dtype))

    def _make_layer(
        self,
        block: Type[Bottleneck],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilation: int = 1,
        norm_layer: Optional[nn.Cell] = None,
        is_first: bool = True,
    ) -> nn.SequentialCell:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            down_layers = []
            if self.avg_down:
                if dilation == 1:
                    down_layers.append(mint.nn.AvgPool2d(kernel_size=stride, stride=stride, padding=0))
                else:
                    down_layers.append(mint.nn.AvgPool2d(kernel_size=1, stride=1, padding=0))

                down_layers.append(
                    mint.nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=1, bias=False)
                )
            else:
                down_layers.append(
                    mint.nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False)
                )
            down_layers.append(norm_layer(planes * block.expansion))
            downsample = nn.SequentialCell(down_layers)

        layers = []
        if dilation == 1 or dilation == 2:
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    stride,
                    downsample=downsample,
                    radix=self.radix,
                    cardinality=self.cardinality,
                    bottleneck_width=self.bottleneck_width,
                    avd=self.avd,
                    avd_first=self.avd_first,
                    dilation=1,
                    is_first=is_first,
                    norm_layer=norm_layer,
                )
            )
        elif dilation == 4:
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    stride,
                    downsample=downsample,
                    radix=self.radix,
                    cardinality=self.cardinality,
                    bottleneck_width=self.bottleneck_width,
                    avd=self.avd,
                    avd_first=self.avd_first,
                    dilation=2,
                    is_first=is_first,
                    norm_layer=norm_layer,
                )
            )
        else:
            raise ValueError(f"Unsupported model type {dilation}")

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    radix=self.radix,
                    cardinality=self.cardinality,
                    bottleneck_width=self.bottleneck_width,
                    avd=self.avd,
                    avd_first=self.avd_first,
                    dilation=dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.SequentialCell(layers)

    def forward_features(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward_head(self, x: Tensor) -> Tensor:
        x = self.avgpool(x)
        if self.drop:
            x = self.drop(x)
        x = self.fc(x)
        return x

    def construct(self, x: Tensor) -> Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _create_resnest(pretrained=False, **kwargs):
    return build_model_with_cfg(ResNeSt, pretrained, **kwargs)


@register_model
def resnest14(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs["resnest14"]
    model_args = dict(block=Bottleneck, layers=[1, 1, 1, 1], radix=2, group=1,
                      bottleneck_width=64, num_classes=num_classes,
                      deep_stem=True, stem_width=32, avg_down=True,
                      avd=True, avd_first=False, **kwargs)
    return _create_resnest(pretrained, **dict(default_cfg=default_cfg, **model_args))


@register_model
def resnest26(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs["resnest26"]
    model_args = dict(block=Bottleneck, layers=[2, 2, 2, 2], radix=2, group=1,
                      bottleneck_width=64, num_classes=num_classes,
                      deep_stem=True, stem_width=32, avg_down=True,
                      avd=True, avd_first=False, **kwargs)
    return _create_resnest(pretrained, **dict(default_cfg=default_cfg, **model_args))


@register_model
def resnest50(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs["resnest50"]
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], radix=2, group=1,
                      bottleneck_width=64, num_classes=num_classes,
                      deep_stem=True, stem_width=32, avg_down=True,
                      avd=True, avd_first=False, **kwargs)
    return _create_resnest(pretrained, **dict(default_cfg=default_cfg, **model_args))


@register_model
def resnest101(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs["resnest101"]
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], radix=2, group=1,
                      bottleneck_width=64, num_classes=num_classes,
                      deep_stem=True, stem_width=64, avg_down=True,
                      avd=True, avd_first=False, **kwargs)
    return _create_resnest(pretrained, **dict(default_cfg=default_cfg, **model_args))


@register_model
def resnest200(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs["resnest200"]
    model_args = dict(block=Bottleneck, layers=[3, 24, 36, 3], radix=2, group=1,
                      bottleneck_width=64, num_classes=num_classes,
                      deep_stem=True, stem_width=64, avg_down=True,
                      avd=True, avd_first=False, **kwargs)
    return _create_resnest(pretrained, **dict(default_cfg=default_cfg, **model_args))


@register_model
def resnest269(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs["resnest269"]
    model_args = dict(block=Bottleneck, layers=[3, 30, 48, 8], radix=2, group=1,
                      bottleneck_width=64, num_classes=num_classes,
                      deep_stem=True, stem_width=64, avg_down=True,
                      avd=True, avd_first=False, **kwargs)
    return _create_resnest(pretrained, **dict(default_cfg=default_cfg, **model_args))
