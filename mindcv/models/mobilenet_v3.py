import math
from typing import Optional, List

import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.common.initializer as init
from mindspore import Tensor

from .layers.conv_norm_act import Conv2dNormActivation
from .layers.squeeze_excite import SqueezeExcite
from .layers.pooling import GlobalAvgPooling
from .utils import load_pretrained, make_divisible
from .registry import register_model


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000,
        'first_conv': '', 'classifier': '',
        **kwargs
    }


default_cfgs = {
    'mobilenet_v3_small_1.0': _cfg(url=''),
    'mobilenet_v3_large_1.0': _cfg(url=''),
    'mobilenet_v3_small_0.75': _cfg(url=''),
    'mobilenet_v3_large_0.75': _cfg(url='')
}


class ResUnit(nn.Cell):

    def __init__(self,
                 in_channels: int,
                 mid_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 norm: nn.Cell,
                 activation: str,
                 stride: int = 1,
                 use_se: bool = False) -> None:
        super(ResUnit, self).__init__()
        self.use_se = use_se
        self.use_short_cut_conv = in_channels == out_channels and stride == 1
        self.use_hs = activation == 'hswish'
        self.activation = nn.HSwish if self.use_hs else nn.ReLU

        layers = []

        # Expand.
        if in_channels != mid_channels:
            layers.append(
                Conv2dNormActivation(in_channels, mid_channels, kernel_size=1, norm=norm, activation=self.activation)
            )

        # DepthWise.
        layers.append(
            Conv2dNormActivation(mid_channels, mid_channels, kernel_size=kernel_size, stride=stride,
                                 groups=mid_channels, norm=norm, activation=self.activation)
        )
        if use_se:
            squeeze_channel = make_divisible(mid_channels // 4, 8)
            layers.append(
                SqueezeExcite(mid_channels,
                              rd_channels=squeeze_channel,
                              act_layer=nn.ReLU,
                              gate_layer=nn.HSigmoid)
            )

        # Project.
        layers.append(
            Conv2dNormActivation(mid_channels, out_channels, kernel_size=1, norm=norm, activation=None)
        )

        self.block = nn.SequentialCell(layers)

    def construct(self, x: Tensor) -> Tensor:
        out = self.block(x)

        if self.use_short_cut_conv:
            out = ops.add(out, x)

        return out


class MobileNetV3(nn.Cell):

    def __init__(self,
                 model_cfgs: List,
                 last_channels: int,
                 in_channels: int = 3,
                 alpha: float = 1.0,
                 norm: Optional[nn.Cell] = None,
                 round_nearest: int = 8,
                 num_classes: int = 1000,
                 drop_rate: float = 0.2
                 ) -> None:
        super(MobileNetV3, self).__init__()

        if not norm:
            norm = nn.BatchNorm2d

        self.input_channels = 16
        layers = []

        # Building first layer.
        first_conv_in_channel = in_channels
        first_conv_out_channel = make_divisible(self.input_channels * alpha, round_nearest)
        layers.append(
            Conv2dNormActivation(
                first_conv_in_channel,
                first_conv_out_channel,
                kernel_size=3,
                stride=2,
                norm=norm,
                activation=nn.HSwish
            )
        )

        # Building inverted residual blocks.
        for layer_cfg in model_cfgs:
            layers.append(self._make_layer(kernel_size=layer_cfg[0],
                                           exp_channels=make_divisible(alpha * layer_cfg[1], round_nearest),
                                           out_channels=make_divisible(alpha * layer_cfg[2], round_nearest),
                                           use_se=layer_cfg[3],
                                           activation=layer_cfg[4],
                                           stride=layer_cfg[5],
                                           norm=norm
                                           )
                          )

        lastconv_input_channel = make_divisible(alpha * model_cfgs[-1][2], round_nearest)
        lastconv_output_channel = lastconv_input_channel * 6

        # Building last several layers.
        layers.append(
            Conv2dNormActivation(
                lastconv_input_channel,
                lastconv_output_channel,
                kernel_size=1,
                norm=norm,
                activation=nn.HSwish
            )
        )

        self.features = nn.SequentialCell(layers)

        self.pool = GlobalAvgPooling(keep_dims=True)
        self.conv_head = nn.SequentialCell([
            nn.Conv2d(in_channels=lastconv_output_channel,
                      out_channels=last_channels,
                      kernel_size=1,
                      stride=1),
            nn.HSwish(),
        ])
        self.dropout = nn.Dropout(keep_prob=1 - drop_rate)
        self.classifier = nn.Conv2d(in_channels=last_channels,
                                    out_channels=num_classes,
                                    kernel_size=1,
                                    has_bias=True)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                n = cell.kernel_size[0] * cell.kernel_size[1] * cell.out_channels
                cell.weight.set_data(
                    init.initializer(init.Normal(sigma=math.sqrt(2. / n), mean=0.0),
                                     cell.weight.shape,
                                     cell.weight.dtype
                                     ),
                )
                if cell.bias is not None:
                    cell.bias.set_data(
                        init.initializer('zeros', cell.bias.shape, cell.bias.dtype)
                    )
            elif isinstance(cell, nn.BatchNorm2d):
                cell.gamma.set_data(
                    init.initializer('ones', cell.gamma.shape, cell.gamma.dtype)
                )
                cell.beta.set_data(
                    init.initializer('zeros', cell.beta.shape, cell.beta.dtype)
                )
            elif isinstance(cell, nn.Dense):
                cell.weight.set_data(
                    init.initializer(init.Normal(sigma=0.01, mean=0.0),
                                     cell.weight.shape,
                                     cell.weight.dtype
                                     )
                )
                if cell.bias is not None:
                    cell.bias.set_data(
                        init.initializer('zeros', cell.bias.shape, cell.bias.dtype)
                    )

    def forward_features(self, x: Tensor) -> Tensor:
        x = self.features(x)
        return x

    def forward_head(self, x: Tensor) -> Tensor:
        x = self.pool(x)
        x = self.conv_head(x)
        x = self.dropout(x)
        x = self.classifier(x)
        x = ops.squeeze(x, axis=(2, 3))
        return x

    def construct(self, x: Tensor) -> Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x

    def _make_layer(self,
                    kernel_size: int,
                    exp_channels: int,
                    out_channels: int,
                    use_se: bool,
                    activation: str,
                    norm: nn.Cell,
                    stride: int = 1
                    ) -> ResUnit:
        """Block layers."""
        layer = ResUnit(self.input_channels, exp_channels, out_channels,
                        kernel_size=kernel_size, stride=stride, activation=activation, use_se=use_se, norm=norm)
        self.input_channels = out_channels

        return layer


model_cfgs = {
    "large": [
        [3, 16, 16, False, 'relu', 1],
        [3, 64, 24, False, 'relu', 2],
        [3, 72, 24, False, 'relu', 1],
        [5, 72, 40, True, 'relu', 2],
        [5, 120, 40, True, 'relu', 1],
        [5, 120, 40, True, 'relu', 1],
        [3, 240, 80, False, 'hswish', 2],
        [3, 200, 80, False, 'hswish', 1],
        [3, 184, 80, False, 'hswish', 1],
        [3, 184, 80, False, 'hswish', 1],
        [3, 480, 112, True, 'hswish', 1],
        [3, 672, 112, True, 'hswish', 1],
        [5, 672, 160, True, 'hswish', 2],
        [5, 960, 160, True, 'hswish', 1],
        [5, 960, 160, True, 'hswish', 1]
    ],
    "small": [
        [3, 16, 16, True, 'relu', 2],
        [3, 72, 24, False, 'relu', 2],
        [3, 88, 24, False, 'relu', 1],
        [5, 96, 40, True, 'hswish', 2],
        [5, 240, 40, True, 'hswish', 1],
        [5, 240, 40, True, 'hswish', 1],
        [5, 120, 48, True, 'hswish', 1],
        [5, 144, 48, True, 'hswish', 1],
        [5, 288, 96, True, 'hswish', 2],
        [5, 576, 96, True, 'hswish', 1],
        [5, 576, 96, True, 'hswish', 1]]
}


@register_model
def mobilenet_v3_small_100(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> MobileNetV3:
    default_cfg = default_cfgs['mobilenet_v3_small_1.0']
    model = MobileNetV3(model_cfgs=model_cfgs['small'],
                        last_channels=1280,
                        in_channels=in_channels,
                        alpha=1.0,
                        num_classes=num_classes,
                        **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def mobilenet_v3_large_100(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> MobileNetV3:
    default_cfg = default_cfgs['mobilenet_v3_large_1.0']
    model = MobileNetV3(model_cfgs=model_cfgs['large'],
                        last_channels=1280,
                        in_channels=in_channels,
                        alpha=1.0,
                        num_classes=num_classes,
                        **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def mobilenet_v3_small_075(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> MobileNetV3:
    default_cfg = default_cfgs['mobilenet_v3_small_0.75']
    model = MobileNetV3(model_cfgs=model_cfgs['small'],
                        last_channels=1280,
                        in_channels=in_channels,
                        alpha=0.75,
                        num_classes=num_classes,
                        **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def mobilenet_v3_large_075(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> MobileNetV3:
    default_cfg = default_cfgs['mobilenet_v3_large_0.75']
    model = MobileNetV3(model_cfgs=model_cfgs['large'],
                        last_channels=1280,
                        in_channels=in_channels,
                        alpha=0.75,
                        num_classes=num_classes,
                        **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model
