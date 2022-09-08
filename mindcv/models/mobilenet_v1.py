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

import mindspore.nn as nn
import mindspore.common.initializer as init
from mindspore import Tensor

from .utils import load_pretrained
from .registry import register_model
from .layers.conv_norm_act import Conv2dNormActivation
from .layers.pooling import GlobalAvgPooling

__all__ = [
    'MobileNetV1',
    'mobilenet_v1_025_224',
    'mobilenet_v1_050_224',
    'mobilenet_v1_075_224',
    'mobilenet_v1_100_224',
]


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000,
        'first_conv': '', 'classifier': '',
        **kwargs
    }


default_cfgs = {
    'mobilenet_v1_0.25_224': _cfg(url=''),
    'mobilenet_v1_0.5_224': _cfg(url=''),
    'mobilenet_v1_0.75_224': _cfg(url=''),
    'mobilenet_v1_1.0_224': _cfg(url=''),

}


class DepthwiseSeparable(nn.Cell):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int
                 ) -> None:
        super(DepthwiseSeparable, self).__init__()

        self.dw_conv = Conv2dNormActivation(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            groups=in_channels)

        self.pw_conv = Conv2dNormActivation(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1)

    def construct(self, x: Tensor) -> Tensor:
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        return x


class MobileNetV1(nn.Cell):

    def __init__(self,
                 alpha: float = 1.0,
                 num_classes: int = 1000,
                 in_channels: int = 3):
        super(MobileNetV1, self).__init__()
        layers = list()

        layers.append(
            Conv2dNormActivation(
                in_channels=in_channels,
                out_channels=int(32 * alpha),
                kernel_size=3,
                stride=2
            )
        )
        layers.append(
            DepthwiseSeparable(
                in_channels=int(32 * alpha),
                out_channels=int(64 * alpha),
                stride=1
            )
        )
        layers.append(
            DepthwiseSeparable(
                in_channels=int(64 * alpha),
                out_channels=int(128 * alpha),
                stride=2
            )
        )
        layers.append(
            DepthwiseSeparable(
                in_channels=int(128 * alpha),
                out_channels=int(128 * alpha),
                stride=1
            )
        )
        layers.append(
            DepthwiseSeparable(
                in_channels=int(128 * alpha),
                out_channels=int(256 * alpha),
                stride=2
            )
        )
        layers.append(
            DepthwiseSeparable(
                in_channels=int(256 * alpha),
                out_channels=int(256 * alpha),
                stride=1
            )
        )
        layers.append(
            DepthwiseSeparable(
                in_channels=int(256 * alpha),
                out_channels=int(512 * alpha),
                stride=2
            )
        )
        for i in range(5):
            layers.append(
                DepthwiseSeparable(
                    in_channels=int(512 * alpha),
                    out_channels=int(512 * alpha),
                    stride=1
                )
            )
        layers.append(
            DepthwiseSeparable(
                in_channels=int(512 * alpha),
                out_channels=int(1024 * alpha),
                stride=2
            )
        )
        layers.append(
            DepthwiseSeparable(
                in_channels=int(1024 * alpha),
                out_channels=int(1024 * alpha),
                stride=1
            )
        )

        self.features = nn.SequentialCell(layers)
        self.pool = GlobalAvgPooling()
        self.classifier = nn.Dense(int(1024 * alpha), num_classes)
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
        return x

    def construct(self, x: Tensor) -> Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


@register_model
def mobilenet_v1_025_224(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> MobileNetV1:
    default_cfg = default_cfgs['mobilenet_v1_0.25_224']
    model = MobileNetV1(alpha=0.25, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def mobilenet_v1_050_224(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> MobileNetV1:
    default_cfg = default_cfgs['mobilenet_v1_0.5_224']
    model = MobileNetV1(alpha=0.5, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def mobilenet_v1_075_224(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> MobileNetV1:
    default_cfg = default_cfgs['mobilenet_v1_0.75_224']
    model = MobileNetV1(alpha=0.75, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def mobilenet_v1_100_224(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> MobileNetV1:
    default_cfg = default_cfgs['mobilenet_v1_1.0_224']
    model = MobileNetV1(alpha=1.0, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model
