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

from typing import Union, Tuple

import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.common.initializer as init
from mindspore import Tensor

from .utils import load_pretrained
from .registry import register_model
from .layers.pooling import GlobalAvgPooling

__all__ = [
    'InceptionV3',
    'inception_v3'
]


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000,
        'first_conv': '', 'classifier': '',
        **kwargs
    }


default_cfgs = {
    'inception_v3': _cfg(url='')
}


class BasicConv2d(nn.Cell):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple] = 1,
                 stride: int = 1,
                 padding: int = 0,
                 pad_mode: str = 'same'
                 ) -> None:
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              pad_mode=pad_mode)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.9997)
        self.relu = nn.ReLU()

    def construct(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class InceptionA(nn.Cell):
    def __init__(self,
                 in_channels: int,
                 pool_features: int) -> None:
        super(InceptionA, self).__init__()
        self.branch0 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch1 = nn.SequentialCell([
            BasicConv2d(in_channels, 48, kernel_size=1),
            BasicConv2d(48, 64, kernel_size=5)
        ])
        self.branch2 = nn.SequentialCell([
            BasicConv2d(in_channels, 64, kernel_size=1),
            BasicConv2d(64, 96, kernel_size=3),
            BasicConv2d(96, 96, kernel_size=3)

        ])
        self.branch_pool = nn.SequentialCell([
            nn.AvgPool2d(kernel_size=3, pad_mode='same'),
            BasicConv2d(in_channels, pool_features, kernel_size=1)
        ])

    def construct(self, x: Tensor) -> Tensor:
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        branch_pool = self.branch_pool(x)
        out = ops.concat((x0, x1, x2, branch_pool), axis=1)
        return out


class InceptionB(nn.Cell):
    def __init__(self, in_channels: int) -> None:
        super(InceptionB, self).__init__()
        self.branch0 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2, pad_mode='valid')
        self.branch1 = nn.SequentialCell([
            BasicConv2d(in_channels, 64, kernel_size=1),
            BasicConv2d(64, 96, kernel_size=3),
            BasicConv2d(96, 96, kernel_size=3, stride=2, pad_mode='valid')

        ])
        self.branch_pool = nn.MaxPool2d(kernel_size=3, stride=2)

    def construct(self, x: Tensor) -> Tensor:
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        branch_pool = self.branch_pool(x)
        out = ops.concat((x0, x1, branch_pool), axis=1)
        return out


class InceptionC(nn.Cell):
    def __init__(self,
                 in_channels: int,
                 channels_7x7: int) -> None:
        super(InceptionC, self).__init__()
        self.branch0 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch1 = nn.SequentialCell([
            BasicConv2d(in_channels, channels_7x7, kernel_size=1),
            BasicConv2d(channels_7x7, channels_7x7, kernel_size=(1, 7)),
            BasicConv2d(channels_7x7, 192, kernel_size=(7, 1))
        ])
        self.branch2 = nn.SequentialCell([
            BasicConv2d(in_channels, channels_7x7, kernel_size=1),
            BasicConv2d(channels_7x7, channels_7x7, kernel_size=(7, 1)),
            BasicConv2d(channels_7x7, channels_7x7, kernel_size=(1, 7)),
            BasicConv2d(channels_7x7, channels_7x7, kernel_size=(7, 1)),
            BasicConv2d(channels_7x7, 192, kernel_size=(1, 7))
        ])
        self.branch_pool = nn.SequentialCell([
            nn.AvgPool2d(kernel_size=3, pad_mode='same'),
            BasicConv2d(in_channels, 192, kernel_size=1)
        ])

    def construct(self, x: Tensor) -> Tensor:
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        branch_pool = self.branch_pool(x)
        out = ops.concat((x0, x1, x2, branch_pool), axis=1)
        return out


class InceptionD(nn.Cell):
    def __init__(self, in_channels: int) -> None:
        super(InceptionD, self).__init__()
        self.branch0 = nn.SequentialCell([
            BasicConv2d(in_channels, 192, kernel_size=1),
            BasicConv2d(192, 320, kernel_size=3, stride=2, pad_mode='valid')
        ])
        self.branch1 = nn.SequentialCell([
            BasicConv2d(in_channels, 192, kernel_size=1),
            BasicConv2d(192, 192, kernel_size=(1, 7)),  # check
            BasicConv2d(192, 192, kernel_size=(7, 1)),
            BasicConv2d(192, 192, kernel_size=3, stride=2, pad_mode='valid')
        ])
        self.branch_pool = nn.MaxPool2d(kernel_size=3, stride=2)

    def construct(self, x: Tensor) -> Tensor:
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        branch_pool = self.branch_pool(x)
        out = ops.concat((x0, x1, branch_pool), axis=1)
        return out


class InceptionE(nn.Cell):
    def __init__(self, in_channels: int) -> None:
        super(InceptionE, self).__init__()
        self.branch0 = BasicConv2d(in_channels, 320, kernel_size=1)
        self.branch1 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch1_a = BasicConv2d(384, 384, kernel_size=(1, 3))
        self.branch1_b = BasicConv2d(384, 384, kernel_size=(3, 1))
        self.branch2 = nn.SequentialCell([
            BasicConv2d(in_channels, 448, kernel_size=1),
            BasicConv2d(448, 384, kernel_size=3)
        ])
        self.branch2_a = BasicConv2d(384, 384, kernel_size=(1, 3))
        self.branch2_b = BasicConv2d(384, 384, kernel_size=(3, 1))
        self.branch_pool = nn.SequentialCell([
            nn.AvgPool2d(kernel_size=3, pad_mode='same'),
            BasicConv2d(in_channels, 192, kernel_size=1)
        ])

    def construct(self, x: Tensor) -> Tensor:
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x1 = ops.concat((self.branch1_a(x1), self.branch1_b(x1)), axis=1)
        x2 = self.branch2(x)
        x2 = ops.concat((self.branch2_a(x2), self.branch2_b(x2)), axis=1)
        branch_pool = self.branch_pool(x)
        out = ops.concat((x0, x1, x2, branch_pool), axis=1)
        return out


class InceptionAux(nn.Cell):
    def __init__(self,
                 in_channels: int,
                 num_classes: int) -> None:
        super(InceptionAux, self).__init__()
        self.avg_pool = nn.AvgPool2d(5, stride=3, pad_mode='valid')
        self.conv2d_0 = BasicConv2d(in_channels, 128, kernel_size=1)
        self.conv2d_1 = BasicConv2d(128, 768, kernel_size=5, pad_mode='valid')
        self.flatten = nn.Flatten()
        self.fc = nn.Dense(in_channels, num_classes)

    def construct(self, x: Tensor) -> Tensor:
        x = self.avg_pool(x)
        x = self.conv2d_0(x)
        x = self.conv2d_1(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class InceptionV3(nn.Cell):
    def __init__(self,
                 num_classes: int = 1000,
                 aux_logits: bool = True,
                 in_channels: int = 3,
                 dropout_rate: float = 0.2) -> None:
        super(InceptionV3, self).__init__()
        self.aux_logits = aux_logits
        self.Conv2d_1a = BasicConv2d(in_channels, 32, kernel_size=3, stride=2, pad_mode='valid')
        self.Conv2d_2a = BasicConv2d(32, 32, kernel_size=3, stride=1, pad_mode='valid')
        self.Conv2d_2b = BasicConv2d(32, 64, kernel_size=3, stride=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Conv2d_3b = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a = BasicConv2d(80, 192, kernel_size=3, pad_mode='valid')
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)
        self.Mixed_6a = InceptionB(288)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)
        if aux_logits:
            self.AuxLogits = InceptionAux(768, num_classes)
        self.Mixed_7a = InceptionD(768)
        self.Mixed_7b = InceptionE(1280)
        self.Mixed_7c = InceptionE(2048)

        self.pool = GlobalAvgPooling()
        self.dropout = nn.Dropout(keep_prob=1 - dropout_rate)
        self.num_features = 2048
        self.classifier = nn.Dense(self.num_features, num_classes)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(
                    init.initializer(init.XavierUniform(),
                                     cell.weight.shape, cell.weight.dtype))

    def forward_preaux(self, x: Tensor) -> Tensor:
        x = self.Conv2d_1a(x)
        x = self.Conv2d_2a(x)
        x = self.Conv2d_2b(x)
        x = self.maxpool1(x)
        x = self.Conv2d_3b(x)
        x = self.Conv2d_4a(x)
        x = self.maxpool2(x)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        return x

    def forward_postaux(self, x: Tensor) -> Tensor:
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)
        return x

    def forward_features(self, x: Tensor) -> Tensor:
        x = self.forward_preaux(x)
        x = self.forward_postaux(x)
        return x

    def construct(self, x: Tensor) -> Tensor:
        x = self.forward_preaux(x)

        if self.training and self.aux_logits:
            aux_logits = self.AuxLogits(x)
        else:
            aux_logits = None

        x = self.forward_postaux(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = self.classifier(x)

        if self.training and self.aux_logits:
            return x, aux_logits
        return x


@register_model
def inception_v3(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> InceptionV3:
    default_cfg = default_cfgs['inception_v3']
    model = InceptionV3(num_classes=num_classes, aux_logits=True, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model
