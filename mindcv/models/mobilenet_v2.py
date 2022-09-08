#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import math
from typing import Optional, List

import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.common.initializer as init
from mindspore import Tensor

from .layers.pooling import GlobalAvgPooling
from .layers.conv_norm_act import Conv2dNormActivation
from .utils import make_divisible, load_pretrained
from .registry import register_model

__all__ = [
    'MobileNetV2',
    'mobilenet_v2_140_224',
    'mobilenet_v2_130_224',
    'mobilenet_v2_100_224',
    'mobilenet_v2_100_192',
    'mobilenet_v2_100_160',
    'mobilenet_v2_100_128',
    'mobilenet_v2_100_96',
    'mobilenet_v2_075_224',
    'mobilenet_v2_075_192',
    'mobilenet_v2_075_160',
    'mobilenet_v2_075_128',
    'mobilenet_v2_075_96',
    'mobilenet_v2_050_224',
    'mobilenet_v2_050_192',
    'mobilenet_v2_050_160',
    'mobilenet_v2_050_128',
    'mobilenet_v2_050_96',
    'mobilenet_v2_035_224',
    'mobilenet_v2_035_192',
    'mobilenet_v2_035_160',
    'mobilenet_v2_035_128',
    'mobilenet_v2_035_96'
]


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000,
        'first_conv': 'features.0.features.0', 'classifier': 'classifier',
        **kwargs
    }


default_cfgs = {
    'mobilenet_v2_1.4_224': _cfg(url=''),
    'mobilenet_v2_1.3_224': _cfg(url=''),
    'mobilenet_v2_1.0_224': _cfg(url=''),
    'mobilenet_v2_1.0_192': _cfg(url=''),
    'mobilenet_v2_1.0_160': _cfg(url=''),
    'mobilenet_v2_1.0_128': _cfg(url=''),
    'mobilenet_v2_1.0_96': _cfg(url=''),
    'mobilenet_v2_0.75_224': _cfg(url=''),
    'mobilenet_v2_0.75_192': _cfg(url=''),
    'mobilenet_v2_0.75_160': _cfg(url=''),
    'mobilenet_v2_0.75_128': _cfg(url=''),
    'mobilenet_v2_0.75_96': _cfg(url=''),
    'mobilenet_v2_0.5_224': _cfg(url=''),
    'mobilenet_v2_0.5_192': _cfg(url=''),
    'mobilenet_v2_0.5_160': _cfg(url=''),
    'mobilenet_v2_0.5_128': _cfg(url=''),
    'mobilenet_v2_0.5_96': _cfg(url=''),
    'mobilenet_v2_0.35_224': _cfg(url=''),
    'mobilenet_v2_0.35_192': _cfg(url=''),
    'mobilenet_v2_0.35_160': _cfg(url=''),
    'mobilenet_v2_0.35_128': _cfg(url=''),
    'mobilenet_v2_0.35_96': _cfg(url=''),
}


class InvertedResidual(nn.Cell):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int,
                 expand_ratio: int,
                 norm: Optional[nn.Cell] = None,
                 last_relu: bool = False
                 ) -> None:
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        if not norm:
            norm = nn.BatchNorm2d

        hidden_dim = round(in_channels * expand_ratio)
        self.use_res_connect = stride == 1 and in_channels == out_channels

        layers: List[nn.Cell] = []
        if expand_ratio != 1:
            # pw
            layers.append(
                Conv2dNormActivation(in_channels, hidden_dim, kernel_size=1, norm=norm, activation=nn.ReLU6)
            )
        layers.extend([
            # dw
            Conv2dNormActivation(
                hidden_dim,
                hidden_dim,
                stride=stride,
                groups=hidden_dim,
                norm=norm,
                activation=nn.ReLU6
            ),
            # pw-linear
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1,
                      stride=1, has_bias=False),
            norm(out_channels)
        ])
        self.conv = nn.SequentialCell(layers)
        self.last_relu = last_relu
        self.relu = nn.ReLU6()

    def construct(self, x: Tensor) -> Tensor:
        identity = x
        x = self.conv(x)
        if self.use_res_connect:
            x = ops.add(identity, x)
        if self.last_relu:
            x = self.relu(x)
        return x


class MobileNetV2(nn.Cell):

    def __init__(self,
                 alpha: float = 1.0,
                 inverted_residual_setting: Optional[List[List[int]]] = None,
                 round_nearest: int = 8,
                 block: Optional[nn.Cell] = None,
                 norm: Optional[nn.Cell] = None,
                 num_classes: int = 1000,
                 in_channels: int = 3
                 ) -> None:
        super(MobileNetV2, self).__init__()

        if not block:
            block = InvertedResidual
        if not norm:
            norm = nn.BatchNorm2d

        input_channel = make_divisible(32 * alpha, round_nearest)
        last_channel = make_divisible(1280 * max(1.0, alpha), round_nearest)

        # Setting of inverted residual blocks.
        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # Building first layer.
        features: List[nn.Cell] = [
            Conv2dNormActivation(in_channels, input_channel, stride=2, norm=norm, activation=nn.ReLU6)
        ]

        # Building inverted residual blocks.
        # t: The expansion factor.
        # c: Number of output channel.
        # n: Number of block.
        # s: First block stride.
        for t, c, n, s in inverted_residual_setting:
            output_channel = make_divisible(c * alpha, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm=norm))
                input_channel = output_channel

        # Building last several layers.
        features.append(
            Conv2dNormActivation(input_channel, last_channel, kernel_size=1, norm=norm, activation=nn.ReLU6)
        )
        # Make it nn.CellList.
        self.features = nn.SequentialCell(features)

        self.pool = GlobalAvgPooling(keep_dims=True)

        self.classifier = nn.Conv2d(last_channel, num_classes, kernel_size=1, has_bias=True)
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
        x = self.classifier(x)
        x = ops.squeeze(x, axis=(2, 3))
        return x

    def construct(self, x: Tensor) -> Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


@register_model
def mobilenet_v2_140_224(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> MobileNetV2:
    default_cfg = default_cfgs['mobilenet_v2_1.4_224']
    model = MobileNetV2(alpha=1.4, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def mobilenet_v2_130_224(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> MobileNetV2:
    default_cfg = default_cfgs['mobilenet_v2_1.3_224']
    model = MobileNetV2(alpha=1.3, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def mobilenet_v2_100_224(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> MobileNetV2:
    default_cfg = default_cfgs['mobilenet_v2_1.0_224']
    model = MobileNetV2(alpha=1.0, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def mobilenet_v2_100_192(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> MobileNetV2:
    default_cfg = default_cfgs['mobilenet_v2_1.0_192']
    model = MobileNetV2(alpha=1.0, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def mobilenet_v2_100_160(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> MobileNetV2:
    default_cfg = default_cfgs['mobilenet_v2_1.0_160']
    model = MobileNetV2(alpha=1.0, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def mobilenet_v2_100_128(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> MobileNetV2:
    default_cfg = default_cfgs['mobilenet_v2_1.0_128']
    model = MobileNetV2(alpha=1.0, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def mobilenet_v2_100_96(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> MobileNetV2:
    default_cfg = default_cfgs['mobilenet_v2_1.0_96']
    model = MobileNetV2(alpha=1.0, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def mobilenet_v2_075_224(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> MobileNetV2:
    default_cfg = default_cfgs['mobilenet_v2_0.75_224']
    model = MobileNetV2(alpha=0.75, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def mobilenet_v2_075_192(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> MobileNetV2:
    default_cfg = default_cfgs['mobilenet_v2_0.75_192']
    model = MobileNetV2(alpha=0.75, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def mobilenet_v2_075_160(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> MobileNetV2:
    default_cfg = default_cfgs['mobilenet_v2_0.75_160']
    model = MobileNetV2(alpha=0.75, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def mobilenet_v2_075_128(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> MobileNetV2:
    default_cfg = default_cfgs['mobilenet_v2_0.75_128']
    model = MobileNetV2(alpha=0.75, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def mobilenet_v2_075_96(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> MobileNetV2:
    default_cfg = default_cfgs['mobilenet_v2_0.75_96']
    model = MobileNetV2(alpha=0.75, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def mobilenet_v2_050_224(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> MobileNetV2:
    default_cfg = default_cfgs['mobilenet_v2_0.5_224']
    model = MobileNetV2(alpha=0.5, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def mobilenet_v2_050_192(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> MobileNetV2:
    default_cfg = default_cfgs['mobilenet_v2_0.5_192']
    model = MobileNetV2(alpha=0.5, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def mobilenet_v2_050_160(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> MobileNetV2:
    default_cfg = default_cfgs['mobilenet_v2_0.5_160']
    model = MobileNetV2(alpha=0.5, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def mobilenet_v2_050_128(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> MobileNetV2:
    default_cfg = default_cfgs['mobilenet_v2_0.5_128']
    model = MobileNetV2(alpha=0.5, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def mobilenet_v2_050_96(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> MobileNetV2:
    default_cfg = default_cfgs['mobilenet_v2_0.5_96']
    model = MobileNetV2(alpha=0.5, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def mobilenet_v2_035_224(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> MobileNetV2:
    default_cfg = default_cfgs['mobilenet_v2_0.35_224']
    model = MobileNetV2(alpha=0.35, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def mobilenet_v2_035_192(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> MobileNetV2:
    default_cfg = default_cfgs['mobilenet_v2_0.35_192']
    model = MobileNetV2(alpha=0.35, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def mobilenet_v2_035_160(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> MobileNetV2:
    default_cfg = default_cfgs['mobilenet_v2_0.35_160']
    model = MobileNetV2(alpha=0.35, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def mobilenet_v2_035_128(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> MobileNetV2:
    default_cfg = default_cfgs['mobilenet_v2_0.35_128']
    model = MobileNetV2(alpha=0.35, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def mobilenet_v2_035_96(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> MobileNetV2:
    default_cfg = default_cfgs['mobilenet_v2_0.35_96']
    model = MobileNetV2(alpha=0.35, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model
