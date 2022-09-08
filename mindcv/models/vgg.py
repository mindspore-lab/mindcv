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

from typing import List, Dict, Union
import math

import mindspore.nn as nn
import mindspore.common.initializer as init
from mindspore import Tensor

from .utils import load_pretrained
from .registry import register_model

__all__ = [
    'VGG',
    'vgg11',
    'vgg13',
    'vgg16',
    'vgg19'

]


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000,
        'first_conv': '', 'classifier': '',
        **kwargs
    }


default_cfgs = {
    'vgg11': _cfg(url=''),
    'vgg13': _cfg(url=''),
    'vgg16': _cfg(url=''),
    'vgg19': _cfg(url='')

}

cfgs: Dict[str, List[Union[str, int]]] = {
    "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "vgg19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def _make_layers(cfg: List[Union[str, int]],
                 batch_norm: bool = False,
                 in_channels: int = 3) -> nn.SequentialCell:
    layers = []

    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels=in_channels,
                               out_channels=v,
                               kernel_size=3,
                               padding=1,
                               pad_mode="pad")
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]

            in_channels = v

    return nn.SequentialCell(layers)


class VGG(nn.Cell):

    def __init__(self,
                 model_name: str,
                 batch_norm: bool = False,
                 num_classes: int = 1000,
                 in_channels: int = 3,
                 drop_rate: float = 0.5) -> None:
        super(VGG, self).__init__()
        cfg = cfgs[model_name]
        self.features = _make_layers(cfg, batch_norm=batch_norm, in_channels=in_channels)
        self.flatten = nn.Flatten()
        self.classifier = nn.SequentialCell([
            nn.Dense(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(keep_prob=1 - drop_rate),
            nn.Dense(4096, 4096),
            nn.ReLU(),
            nn.Dropout(keep_prob=1 - drop_rate),
            nn.Dense(4096, num_classes),
        ])
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(
                    init.initializer(init.HeNormal(math.sqrt(5), mode='fan_out', nonlinearity='relu'),
                                     cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(
                        init.initializer('zeros', cell.bias.shape, cell.bias.dtype))
            elif isinstance(cell, nn.Dense):
                cell.weight.set_data(
                    init.initializer(init.Normal(0.01), cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(init.initializer('zeros', cell.bias.shape, cell.bias.dtype))

    def forward_features(self, x: Tensor) -> Tensor:
        x = self.features(x)
        return x

    def forward_head(self, x: Tensor) -> Tensor:
        x = self.flatten(x)
        x = self.classifier(x)
        return x

    def construct(self, x: Tensor) -> Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


@register_model
def vgg11(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> VGG:
    default_cfg = default_cfgs['vgg11']
    model = VGG(model_name='vgg11', num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def vgg13(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> VGG:
    default_cfg = default_cfgs['vgg13']
    model = VGG(model_name='vgg13', num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def vgg16(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> VGG:
    default_cfg = default_cfgs['vgg16']
    model = VGG(model_name='vgg16', num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def vgg19(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> VGG:
    default_cfg = default_cfgs['vgg19']
    model = VGG(model_name='vgg19', num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model
