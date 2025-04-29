"""
MindSpore implementation of `MobileNetV3`.
Refer to Searching for MobileNetV3.
"""

import math

import mindspore.common.initializer as init
from mindspore import Tensor, mint, nn

from .helpers import build_model_with_cfg, make_divisible
from .layers.pooling import GlobalAvgPooling
from .layers.squeeze_excite import SqueezeExcite
from .registry import register_model

__all__ = [
    "MobileNetV3",
    "mobilenet_v3_large_075",
    "mobilenet_v3_large_100",
    "mobilenet_v3_small_075",
    "mobilenet_v3_small_100",
]


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "first_conv": "features.0",
        "classifier": "classifier.3",
        **kwargs,
    }


default_cfgs = {
    "mobilenet_v3_small_100": _cfg(
        url="https://download.mindspore.cn/toolkits/mindcv/mobilenet/mobilenetv3/mobilenet_v3_small_100-509c6047.ckpt"
    ),
    "mobilenet_v3_large_100": _cfg(
        url="https://download.mindspore.cn/toolkits/mindcv/mobilenet/mobilenetv3/mobilenet_v3_large_100-1279ad5f.ckpt"
    ),
    "mobilenet_v3_small_075": _cfg(url=""),
    "mobilenet_v3_large_075": _cfg(url=""),
}


class Bottleneck(nn.Cell):
    """Bottleneck Block of MobilenetV3. depth-wise separable convolutions + inverted residual + squeeze excitation"""

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        activation: str = "relu",
        use_se: bool = False,
    ) -> None:
        super().__init__()
        self.use_se = use_se
        self.use_res_connect = stride == 1 and in_channels == out_channels
        assert activation in ["relu", "hswish"]
        self.activation = mint.nn.Hardswish if activation == "hswish" else mint.nn.ReLU

        layers = []
        # Expand.
        if in_channels != mid_channels:
            layers.extend([
                mint.nn.Conv2d(in_channels, mid_channels, 1, 1, padding=0, bias=False),
                mint.nn.BatchNorm2d(mid_channels),
                self.activation(),
            ])
        # DepthWise.
        layers.extend([
            mint.nn.Conv2d(mid_channels, mid_channels, kernel_size, stride,
                      padding=kernel_size // 2, groups=mid_channels, bias=False),
            mint.nn.BatchNorm2d(mid_channels),
            self.activation(),
        ])
        # SqueezeExcitation.
        if use_se:
            layers.append(
                SqueezeExcite(mid_channels, 1.0 / 4, act_layer=mint.nn.ReLU, gate_layer=mint.nn.Hardsigmoid)
            )
        # Project.
        layers.extend([
            mint.nn.Conv2d(mid_channels, out_channels, 1, 1, padding=0, bias=False),
            mint.nn.BatchNorm2d(out_channels),
        ])
        self.layers = nn.SequentialCell(layers)

    def construct(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.layers(x)
        return self.layers(x)


class MobileNetV3(nn.Cell):
    r"""MobileNetV3 model class, based on
    `"Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>`_

    Args:
        arch: size of the architecture. 'small' or 'large'.
        alpha: scale factor of model width. Default: 1.
        round_nearest: divisor of make divisible function. Default: 8.
        in_channels: number the channels of the input. Default: 3.
        num_classes: number of classification classes. Default: 1000.
    """

    def __init__(
        self,
        arch: str,
        alpha: float = 1.0,
        round_nearest: int = 8,
        in_channels: int = 3,
        num_classes: int = 1000,
    ) -> None:
        super().__init__()
        input_channels = make_divisible(16 * alpha, round_nearest)
        # Setting of bottleneck blocks. ex: [k, e, c, se, nl, s]
        # k: kernel size of depth-wise conv
        # e: expansion size
        # c: number of output channel
        # se: whether there is a Squeeze-And-Excite in that block
        # nl: type of non-linearity used
        # s: stride of depth-wise conv
        if arch == "large":
            bottleneck_setting = [
                [3, 16, 16, False, "relu", 1],
                [3, 64, 24, False, "relu", 2],
                [3, 72, 24, False, "relu", 1],
                [5, 72, 40, True, "relu", 2],
                [5, 120, 40, True, "relu", 1],
                [5, 120, 40, True, "relu", 1],
                [3, 240, 80, False, "hswish", 2],
                [3, 200, 80, False, "hswish", 1],
                [3, 184, 80, False, "hswish", 1],
                [3, 184, 80, False, "hswish", 1],
                [3, 480, 112, True, "hswish", 1],
                [3, 672, 112, True, "hswish", 1],
                [5, 672, 160, True, "hswish", 2],
                [5, 960, 160, True, "hswish", 1],
                [5, 960, 160, True, "hswish", 1],
            ]
            last_channels = make_divisible(alpha * 1280, round_nearest)
        elif arch == "small":
            bottleneck_setting = [
                [3, 16, 16, True, "relu", 2],
                [3, 72, 24, False, "relu", 2],
                [3, 88, 24, False, "relu", 1],
                [5, 96, 40, True, "hswish", 2],
                [5, 240, 40, True, "hswish", 1],
                [5, 240, 40, True, "hswish", 1],
                [5, 120, 48, True, "hswish", 1],
                [5, 144, 48, True, "hswish", 1],
                [5, 288, 96, True, "hswish", 2],
                [5, 576, 96, True, "hswish", 1],
                [5, 576, 96, True, "hswish", 1],
            ]
            last_channels = make_divisible(alpha * 1024, round_nearest)
        else:
            raise ValueError(f"Unsupported model type {arch}")

        # Building stem conv layer.
        features = [
            mint.nn.Conv2d(in_channels, input_channels, 3, 2, padding=1, bias=False),
            mint.nn.BatchNorm2d(input_channels),
            mint.nn.Hardswish(),
        ]

        total_reduction = 2
        self.feature_info = [dict(chs=input_channels, reduction=total_reduction, name=f'features.{len(features) - 1}')]

        # Building bottleneck blocks.
        for k, e, c, se, nl, s in bottleneck_setting:
            exp_channels = make_divisible(alpha * e, round_nearest)
            output_channels = make_divisible(alpha * c, round_nearest)
            features.append(Bottleneck(input_channels, exp_channels, output_channels,
                                       kernel_size=k, stride=s, activation=nl, use_se=se))
            input_channels = output_channels

            total_reduction *= s
            self.feature_info.append(dict(chs=input_channels, reduction=total_reduction,
                                          name=f'features.{len(features) - 1}'))

        # Building last point-wise conv layers.
        output_channels = input_channels * 6
        features.extend([
            mint.nn.Conv2d(input_channels, output_channels, 1, 1, padding=0, bias=False),
            mint.nn.BatchNorm2d(output_channels),
            mint.nn.Hardswish(),
        ])

        self.feature_info.append(dict(chs=output_channels, reduction=total_reduction,
                                      name=f'features.{len(features) - 1}'))
        self.flatten_sequential = True

        self.features = nn.SequentialCell(features)

        self.pool = GlobalAvgPooling()
        self.classifier = nn.SequentialCell([
            mint.nn.Linear(output_channels, last_channels),
            mint.nn.Hardswish(),
            mint.nn.Dropout(p=0.2),
            mint.nn.Linear(last_channels, num_classes),
        ])
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize weights for cells."""
        for _, cell in self.cells_and_names():
            if isinstance(cell, mint.nn.Conv2d):
                n = cell.kernel_size[0] * cell.kernel_size[1] * cell.out_channels
                cell.weight.set_data(
                    init.initializer(init.Normal(sigma=math.sqrt(2. / n), mean=0.0),
                                     cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(init.initializer("zeros", cell.bias.shape, cell.bias.dtype))
            elif isinstance(cell, mint.nn.BatchNorm2d):
                cell.weight.set_data(init.initializer("ones", cell.weight.shape, cell.weight.dtype))
                cell.bias.set_data(init.initializer("zeros", cell.bias.shape, cell.bias.dtype))
            elif isinstance(cell, mint.nn.Linear):
                cell.weight.set_data(
                    init.initializer(init.Normal(sigma=0.01, mean=0.0), cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(init.initializer("zeros", cell.bias.shape, cell.bias.dtype))

    def forward_features(self, x: Tensor) -> Tensor:
        x = self.features(x)
        return x

    def forward_head(self, x: Tensor) -> Tensor:
        x = self.pool(x)
        x = self.classifier(x)
        return x

    def construct(self, x: Tensor) -> Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _create_mobilenet_v3(pretrained=False, **kwargs):
    return build_model_with_cfg(MobileNetV3, pretrained, **kwargs)


@register_model
def mobilenet_v3_small_100(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> MobileNetV3:
    """Get small MobileNetV3 model without width scaling.
    Refer to the base class `models.MobileNetV3` for more details.
    """
    default_cfg = default_cfgs["mobilenet_v3_small_100"]
    model_args = dict(arch="small", alpha=1.0, in_channels=in_channels, num_classes=num_classes, **kwargs)
    return _create_mobilenet_v3(pretrained, **dict(default_cfg=default_cfg, **model_args))


@register_model
def mobilenet_v3_large_100(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> MobileNetV3:
    """Get large MobileNetV3 model without width scaling.
    Refer to the base class `models.MobileNetV3` for more details.
    """
    default_cfg = default_cfgs["mobilenet_v3_large_100"]
    model_args = dict(arch="large", alpha=1.0, in_channels=in_channels, num_classes=num_classes, **kwargs)
    return _create_mobilenet_v3(pretrained, **dict(default_cfg=default_cfg, **model_args))


@register_model
def mobilenet_v3_small_075(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> MobileNetV3:
    """Get small MobileNetV3 model with width scaled by 0.75.
    Refer to the base class `models.MobileNetV3` for more details.
    """
    default_cfg = default_cfgs["mobilenet_v3_small_075"]
    model_args = dict(arch="small", alpha=0.75, in_channels=in_channels, num_classes=num_classes, **kwargs)
    return _create_mobilenet_v3(pretrained, **dict(default_cfg=default_cfg, **model_args))


@register_model
def mobilenet_v3_large_075(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> MobileNetV3:
    """Get large MobileNetV3 model with width scaled by 0.75.
    Refer to the base class `models.MobileNetV3` for more details.
    """
    default_cfg = default_cfgs["mobilenet_v3_large_075"]
    model_args = dict(arch="large", alpha=0.75, in_channels=in_channels, num_classes=num_classes, **kwargs)
    return _create_mobilenet_v3(pretrained, **dict(default_cfg=default_cfg, **model_args))
