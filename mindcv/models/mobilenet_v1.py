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


def conv_bn_relu(in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 depthwise: bool,
                 activation: str = 'relu6') -> nn.SequentialCell:
    output = []
    output.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad_mode="same",
                            group=1 if not depthwise else in_channels))
    output.append(nn.BatchNorm2d(out_channels))
    if activation:
        output.append(nn.get_activation(activation))
    return nn.SequentialCell(output)


class MobileNetV1(nn.Cell):
    def __init__(self,
                 alpha: float = 1.,
                 num_classes: int = 1000,
                 in_channels: int = 3) -> None:
        super(MobileNetV1, self).__init__()
        self.layers = [
            conv_bn_relu(in_channels, int(32 * alpha), 3, 2, False),  # Conv0
            conv_bn_relu(int(32 * alpha), int(32 * alpha), 3, 1, True),  # Conv1_depthwise
            conv_bn_relu(int(32 * alpha), int(64 * alpha), 1, 1, False),  # Conv1_pointwise
            conv_bn_relu(int(64 * alpha), int(64 * alpha), 3, 2, True),  # Conv2_depthwise
            conv_bn_relu(int(64 * alpha), int(128 * alpha), 1, 1, False),  # Conv2_pointwise

            conv_bn_relu(int(128 * alpha), int(128 * alpha), 3, 1, True),  # Conv3_depthwise
            conv_bn_relu(int(128 * alpha), int(128 * alpha), 1, 1, False),  # Conv3_pointwise
            conv_bn_relu(int(128 * alpha), int(128 * alpha), 3, 2, True),  # Conv4_depthwise
            conv_bn_relu(int(128 * alpha), int(256 * alpha), 1, 1, False),  # Conv4_pointwise

            conv_bn_relu(int(256 * alpha), int(256 * alpha), 3, 1, True),  # Conv5_depthwise
            conv_bn_relu(int(256 * alpha), int(256 * alpha), 1, 1, False),  # Conv5_pointwise
            conv_bn_relu(int(256 * alpha), int(256 * alpha), 3, 2, True),  # Conv6_depthwise
            conv_bn_relu(int(256 * alpha), int(512 * alpha), 1, 1, False),  # Conv6_pointwise

            conv_bn_relu(int(512 * alpha), int(512 * alpha), 3, 1, True),  # Conv7_depthwise
            conv_bn_relu(int(512 * alpha), int(512 * alpha), 1, 1, False),  # Conv7_pointwise
            conv_bn_relu(int(512 * alpha), int(512 * alpha), 3, 1, True),  # Conv8_depthwise
            conv_bn_relu(int(512 * alpha), int(512 * alpha), 1, 1, False),  # Conv8_pointwise
            conv_bn_relu(int(512 * alpha), int(512 * alpha), 3, 1, True),  # Conv9_depthwise
            conv_bn_relu(int(512 * alpha), int(512 * alpha), 1, 1, False),  # Conv9_pointwise
            conv_bn_relu(int(512 * alpha), int(512 * alpha), 3, 1, True),  # Conv10_depthwise
            conv_bn_relu(int(512 * alpha), int(512 * alpha), 1, 1, False),  # Conv10_pointwise
            conv_bn_relu(int(512 * alpha), int(512 * alpha), 3, 1, True),  # Conv11_depthwise
            conv_bn_relu(int(512 * alpha), int(512 * alpha), 1, 1, False),  # Conv11_pointwise

            conv_bn_relu(int(512 * alpha), int(512 * alpha), 3, 2, True),  # Conv12_depthwise
            conv_bn_relu(int(512 * alpha), int(1024 * alpha), 1, 1, False),  # Conv12_pointwise
            conv_bn_relu(int(1024 * alpha), int(1024 * alpha), 3, 1, True),  # Conv13_depthwise
            conv_bn_relu(int(1024 * alpha), int(1024 * alpha), 1, 1, False),  # Conv13_pointwise

        ]

        self.features = nn.SequentialCell(self.layers)
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
