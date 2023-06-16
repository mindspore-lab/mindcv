"""MindSpore implementation of `GhostNet`.
Refer to GhostNet: More Features from Cheap Operations.
"""

import math

import mindspore.common.initializer as init
from mindspore import Tensor, nn, ops

from .helpers import load_pretrained, make_divisible
from .layers.compatibility import Dropout
from .layers.pooling import GlobalAvgPooling
from .layers.squeeze_excite import SqueezeExcite
from .registry import register_model

__all__ = [
    "GhostNet",
    "ghostnet_050",
    "ghostnet_100",
    "ghostnet_130",
]


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "first_conv": "conv_stem",
        "classifier": "classifier",
        **kwargs,
    }


default_cfgs = {
    "ghostnet_050": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/ghostnet/ghostnet_050-85b91860.ckpt"),
    "ghostnet_100": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/ghostnet/ghostnet_100-bef8025a.ckpt"),
    "ghostnet_130": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/ghostnet/ghostnet_130-cf4c235c.ckpt"),
}


class HardSigmoid(nn.Cell):
    """Implementation for (relu6 + 3) / 6"""

    def __init__(self) -> None:
        super().__init__()
        self.relu6 = nn.ReLU6()

    def construct(self, x: Tensor) -> Tensor:
        return self.relu6(x + 3.0) / 6.0


class ConvBnAct(nn.Cell):
    def __init__(
        self,
        in_chs: int,
        out_chs: int,
        kernel_size: int,
        stride: int = 1,
        act_layer: nn.Cell = nn.ReLU,
    ) -> None:
        super().__init__()
        self.features = nn.SequentialCell([
            nn.Conv2d(in_chs, out_chs, kernel_size, stride, pad_mode="same"),
            nn.BatchNorm2d(out_chs),
            act_layer(),
        ])

    def construct(self, x: Tensor) -> Tensor:
        x = self.features(x)
        return x


class GhostModule(nn.Cell):
    def __init__(
        self,
        inp: int,
        oup: int,
        kernel_size: int = 1,
        ratio: int = 2,
        dw_size: int = 3,
        stride: int = 1,
        relu: bool = True,
    ) -> None:
        super().__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.SequentialCell(
            nn.Conv2d(inp, init_channels, kernel_size, stride, pad_mode="pad",
                      padding=kernel_size // 2, has_bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU() if relu else nn.SequentialCell(),
        )

        self.cheap_operation = nn.SequentialCell(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, pad_mode="pad",
                      padding=dw_size // 2, group=init_channels, has_bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU() if relu else nn.SequentialCell(),
        )

    def construct(self, x: Tensor) -> Tensor:
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = ops.concat((x1, x2), axis=1)
        return out[:, :self.oup, :, :]


class GhostBottleneck(nn.Cell):
    def __init__(
        self,
        in_chs: int,
        mid_chs: int,
        out_chs: int,
        dw_kernel_size: int = 3,
        stride: int = 1,
        se_ratio: float = 0.0,
    ) -> None:
        super().__init__()
        has_se = se_ratio is not None and se_ratio > 0.0
        self.stride = stride

        # Point-wise expansion
        self.ghost1 = GhostModule(in_chs, mid_chs, relu=True)

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride,
                                     pad_mode="pad", padding=(dw_kernel_size - 1) // 2,
                                     group=mid_chs, has_bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_chs, rd_ratio=se_ratio, rd_divisor=4, gate_layer=HardSigmoid)
        else:
            self.se = None

        # Point-wise linear projection
        self.ghost2 = GhostModule(mid_chs, out_chs, relu=False)

        # shortcut
        if (in_chs == out_chs and self.stride == 1):
            self.shortcut = nn.SequentialCell()
        else:
            self.shortcut = nn.SequentialCell(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride, pad_mode="pad",
                          padding=(dw_kernel_size - 1) // 2, group=in_chs, has_bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, pad_mode="pad", padding=0, has_bias=False),
                nn.BatchNorm2d(out_chs),
            )

    def construct(self, x: Tensor) -> Tensor:
        residual = x

        # 1st ghost bottleneck
        x = self.ghost1(x)

        # Depth-wise convolution
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)

        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)

        # 2nd ghost bottleneck
        x = self.ghost2(x)

        x += self.shortcut(residual)
        return x


class GhostNet(nn.Cell):
    r"""GhostNet model class, based on
    `"GhostNet: More Features from Cheap Operations " <https://arxiv.org/abs/1911.11907>`_.
    Args:
        num_classes: number of classification classes. Default: 1000.
        width: base width of hidden channel in blocks. Default: 1.0.
        in_channels: number of input channels. Default: 3.
        drop_rate: the probability of the features before classification. Default: 0.2.
    """

    def __init__(
        self,
        num_classes: int = 1000,
        width: float = 1.0,
        in_channels: int = 3,
        drop_rate: float = 0.2,
    ) -> None:
        super().__init__()
        # setting of inverted residual blocks
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.cfgs = [
            # k, t, c, SE, s
            # stage1
            [[3, 16, 16, 0, 1]],
            # stage2
            [[3, 48, 24, 0, 2]],
            [[3, 72, 24, 0, 1]],
            # stage3
            [[5, 72, 40, 0.25, 2]],
            [[5, 120, 40, 0.25, 1]],
            # stage4
            [[3, 240, 80, 0, 2]],
            [[3, 200, 80, 0, 1],
             [3, 184, 80, 0, 1],
             [3, 184, 80, 0, 1],
             [3, 480, 112, 0.25, 1],
             [3, 672, 112, 0.25, 1]
             ],
            # stage5
            [[5, 672, 160, 0.25, 2]],
            [[5, 960, 160, 0, 1],
             [5, 960, 160, 0.25, 1],
             [5, 960, 160, 0, 1],
             [5, 960, 160, 0.25, 1]
             ]
        ]

        # building first layer
        stem_chs = make_divisible(16 * width, 4)
        self.conv_stem = nn.Conv2d(in_channels, stem_chs, 3, 2, pad_mode="pad", padding=1, has_bias=False)
        self.bn1 = nn.BatchNorm2d(stem_chs)
        self.act1 = nn.ReLU()
        prev_chs = stem_chs

        # building inverted residual blocks
        stages = []
        for cfg in self.cfgs:
            layers = []
            for k, exp_size, c, se_ratio, s in cfg:
                out_chs = make_divisible(c * width, 4)
                mid_chs = make_divisible(exp_size * width, 4)
                layers.append(GhostBottleneck(prev_chs, mid_chs, out_chs, k, s, se_ratio=se_ratio))
                prev_chs = out_chs
            stages.append(nn.SequentialCell(layers))

        out_chs = make_divisible(exp_size * width, 4)
        stages.append(ConvBnAct(prev_chs, out_chs, 1))
        prev_chs = out_chs

        self.blocks = nn.SequentialCell(stages)

        # building last several layers
        self.num_features = out_chs = 1280
        self.global_pool = GlobalAvgPooling(keep_dims=True)
        self.conv_head = nn.Conv2d(prev_chs, out_chs, 1, 1, pad_mode="pad", padding=0, has_bias=True)
        self.act2 = nn.ReLU()
        self.flatten = nn.Flatten()
        if self.drop_rate > 0.0:
            self.dropout = Dropout(p=drop_rate)
        self.classifier = nn.Dense(out_chs, num_classes)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize weights for cells."""
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(init.initializer(init.HeUniform(), cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(init.initializer("zeros", cell.bias.shape, cell.bias.dtype))
            elif isinstance(cell, nn.BatchNorm2d):
                cell.gamma.set_data(init.initializer("ones", cell.gamma.shape, cell.gamma.dtype))
                cell.beta.set_data(init.initializer("zeros", cell.beta.shape, cell.beta.dtype))
            elif isinstance(cell, nn.Dense):
                cell.weight.set_data(init.initializer(init.HeUniform(), cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(init.initializer("zeros", cell.bias.shape, cell.bias.dtype))

    def forward_features(self, x: Tensor) -> Tensor:
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        return x

    def forward_head(self, x: Tensor) -> Tensor:
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act2(x)
        x = self.flatten(x)
        if self.drop_rate > 0.0:
            x = self.dropout(x)
        x = self.classifier(x)
        return x

    def construct(self, x: Tensor) -> Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


@register_model
def ghostnet_050(pretrained: bool = False, num_classes: int = 1000, in_channels: int = 3, **kwargs):
    """ GhostNet-0.5x """
    default_cfg = default_cfgs["ghostnet_050"]
    model = GhostNet(width=0.5, in_channels=in_channels, num_classes=num_classes, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def ghostnet_100(pretrained: bool = False, num_classes: int = 1000, in_channels: int = 3, **kwargs):
    """ GhostNet-1.0x """
    default_cfg = default_cfgs["ghostnet_100"]
    model = GhostNet(width=1.0, in_channels=in_channels, num_classes=num_classes, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def ghostnet_130(pretrained: bool = False, num_classes: int = 1000, in_channels: int = 3, **kwargs):
    """ GhostNet-1.3x """
    default_cfg = default_cfgs["ghostnet_130"]
    model = GhostNet(width=1.3, in_channels=in_channels, num_classes=num_classes, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model
