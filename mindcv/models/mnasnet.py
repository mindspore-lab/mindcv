"""
MindSpore implementation of `MnasNet`.
Refer to MnasNet: Platform-Aware Neural Architecture Search for Mobile.
Adapted from https://github.com/pytorch/vision/blob/main/torchvision/models/mnasnet.py
"""

from typing import List

import mindspore.common.initializer as init
from mindspore import Tensor, nn

from .helpers import load_pretrained, make_divisible
from .layers.compatibility import Dropout
from .layers.pooling import GlobalAvgPooling
from .registry import register_model

__all__ = [
    "Mnasnet",
    "mnasnet_050",
    "mnasnet_075",
    "mnasnet_100",
    "mnasnet_130",
    "mnasnet_140",
]


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "first_conv": "features.0",
        "classifier": "classifier",
        **kwargs,
    }


default_cfgs = {
    "mnasnet_050": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/mnasnet/mnasnet_050-7d8bf4db.ckpt"),
    "mnasnet_075": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/mnasnet/mnasnet_075-465d366d.ckpt"),
    "mnasnet_100": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/mnasnet/mnasnet_100-1bcf43f8.ckpt"),
    "mnasnet_130": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/mnasnet/mnasnet_130-a43a150a.ckpt"),
    "mnasnet_140": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/mnasnet/mnasnet_140-7e20bb30.ckpt"),
}


class InvertedResidual(nn.Cell):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        kernel_size: int,
        expand_ratio: int,
    ) -> None:
        super().__init__()
        assert stride in [1, 2]

        hidden_dim = round(in_channels * expand_ratio)
        self.use_res_connect = stride == 1 and in_channels == out_channels

        self.layers = nn.SequentialCell([
            # pw
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1),
            nn.BatchNorm2d(hidden_dim, momentum=0.99, eps=1e-3),
            nn.ReLU(),
            # dw
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, pad_mode="pad",
                      padding=kernel_size // 2, group=hidden_dim),
            nn.BatchNorm2d(hidden_dim, momentum=0.99, eps=1e-3),
            nn.ReLU(),
            # pw-linear
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels, momentum=0.99, eps=1e-3),
        ])

    def construct(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return self.layers(x) + x
        return self.layers(x)


class Mnasnet(nn.Cell):
    r"""MnasNet model architecture from
    `"MnasNet: Platform-Aware Neural Architecture Search for Mobile" <https://arxiv.org/abs/1807.11626>`_.

    Args:
        alpha: scale factor of model width.
        in_channels: number the channels of the input. Default: 3.
        num_classes: number of classification classes. Default: 1000.
        drop_rate: dropout rate of the layer before main classifier. Default: 0.2.
    """

    def __init__(
        self,
        alpha: float,
        in_channels: int = 3,
        num_classes: int = 1000,
        drop_rate: float = 0.2,
    ):
        super().__init__()

        inverted_residual_setting = [
            # t, c, n, s, k
            [3, 24, 3, 2, 3],  # -> 56x56
            [3, 40, 3, 2, 5],  # -> 28x28
            [6, 80, 3, 2, 5],  # -> 14x14
            [6, 96, 2, 1, 3],  # -> 14x14
            [6, 192, 4, 2, 5],  # -> 7x7
            [6, 320, 1, 1, 3],  # -> 7x7
        ]

        mid_channels = make_divisible(32 * alpha, 8)
        input_channels = make_divisible(16 * alpha, 8)

        features: List[nn.Cell] = [
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=2, pad_mode="pad", padding=1),
            nn.BatchNorm2d(mid_channels, momentum=0.99, eps=1e-3),
            nn.ReLU(),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, pad_mode="pad", padding=1,
                      group=mid_channels),
            nn.BatchNorm2d(mid_channels, momentum=0.99, eps=1e-3),
            nn.ReLU(),
            nn.Conv2d(mid_channels, input_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(input_channels, momentum=0.99, eps=1e-3),
        ]

        for t, c, n, s, k in inverted_residual_setting:
            output_channels = make_divisible(c * alpha, 8)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(input_channels, output_channels,
                                                 stride=stride, kernel_size=k, expand_ratio=t))
                input_channels = output_channels

        features.extend([
            nn.Conv2d(input_channels, 1280, kernel_size=1, stride=1),
            nn.BatchNorm2d(1280, momentum=0.99, eps=1e-3),
            nn.ReLU(),
        ])
        self.features = nn.SequentialCell(features)
        self.pool = GlobalAvgPooling()
        self.dropout = Dropout(p=drop_rate)
        self.classifier = nn.Dense(1280, num_classes)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize weights for cells."""
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(
                    init.initializer(init.HeNormal(mode="fan_out", nonlinearity="relu"),
                                     cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(init.initializer("zeros", cell.bias.shape, cell.bias.dtype))
            elif isinstance(cell, nn.BatchNorm2d):
                cell.gamma.set_data(init.initializer("ones", cell.gamma.shape, cell.gamma.dtype))
                cell.beta.set_data(init.initializer("zeros", cell.beta.shape, cell.beta.dtype))
            elif isinstance(cell, nn.Dense):
                cell.weight.set_data(
                    init.initializer(init.HeUniform(mode="fan_out", nonlinearity="sigmoid"),
                                     cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(init.initializer("zeros", cell.bias.shape, cell.bias.dtype))

    def forward_features(self, x: Tensor) -> Tensor:
        x = self.features(x)
        return x

    def forward_head(self, x: Tensor) -> Tensor:
        x = self.pool(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

    def construct(self, x: Tensor) -> Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


@register_model
def mnasnet_050(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> Mnasnet:
    """Get MnasNet model with width scaled by 0.5.
    Refer to the base class `models.Mnasnet` for more details."""
    default_cfg = default_cfgs["mnasnet_050"]
    model = Mnasnet(alpha=0.5, in_channels=in_channels, num_classes=num_classes, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def mnasnet_075(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> Mnasnet:
    """Get MnasNet model with width scaled by 0.75.
    Refer to the base class `models.Mnasnet` for more details."""
    default_cfg = default_cfgs["mnasnet_075"]
    model = Mnasnet(alpha=0.75, in_channels=in_channels, num_classes=num_classes, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def mnasnet_100(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> Mnasnet:
    """Get MnasNet model with width scaled by 1.0.
    Refer to the base class `models.Mnasnet` for more details."""
    default_cfg = default_cfgs["mnasnet_100"]
    model = Mnasnet(alpha=1.0, in_channels=in_channels, num_classes=num_classes, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def mnasnet_130(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> Mnasnet:
    """Get MnasNet model with width scaled by 1.3.
    Refer to the base class `models.Mnasnet` for more details."""
    default_cfg = default_cfgs["mnasnet_130"]
    model = Mnasnet(alpha=1.3, in_channels=in_channels, num_classes=num_classes, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def mnasnet_140(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> Mnasnet:
    """Get MnasNet model with width scaled by 1.4.
    Refer to the base class `models.Mnasnet` for more details."""
    default_cfg = default_cfgs["mnasnet_140"]
    model = Mnasnet(alpha=1.4, in_channels=in_channels, num_classes=num_classes, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model
