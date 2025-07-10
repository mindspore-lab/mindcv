"""Pre-Activation ResNet v2 with GroupNorm and Weight Standardization.

A PyTorch implementation of ResNetV2 adapted from the Google Big-Transfer (BiT) source code
at https://github.com/google-research/big_transfer to match timm interfaces. The BiT weights have
been included here as pretrained models from their original .NPZ checkpoints.

Additionally, supports non pre-activation bottleneck for use as a backbone for Vision Transformers (ViT) and
extra padding support to allow porting of official Hybrid ResNet pretrained weights from
https://github.com/google-research/vision_transformer

Thanks to the Google team for the above two repositories and associated papers:
* Big Transfer (BiT): General Visual Representation Learning - https://arxiv.org/abs/1912.11370
* An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale - https://arxiv.org/abs/2010.11929
* Knowledge distillation: A good teacher is patient and consistent - https://arxiv.org/abs/2106.05237

Original copyright of Google code below, modifications by Ross Wightman, Copyright 2020.

------------------------------------------------------------------------
MindSpore adaptation:
Modified for use with the MindSpore framework.
This file is part of a derivative work and remains under the original license.
Adapted from https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/resnetv2.py

"""
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

from mindspore import Tensor, nn

from .helpers import load_pretrained
from .registry import register_model
from .resnet import ResNet

__all__ = [
    "resnetv2_50",
    "resnetv2_101",
]


def _cfg(url='', **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "first_conv": "conv1",
        "classifier": "classifier",
        **kwargs
    }


default_cfgs = {
    "resnetv2_50": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/resnetv2/resnetv2_50-3c2f143b.ckpt"),
    "resnetv2_101": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/resnetv2/resnetv2_101-5d4c49a1.ckpt"),
}


class PreActBottleneck(nn.Cell):
    expansion: int = 4

    def __init__(self,
                 in_channels: int,
                 channels: int,
                 stride: int = 1,
                 groups: int = 1,
                 base_width: int = 64,
                 norm: Optional[nn.Cell] = None,
                 down_sample: Optional[nn.Cell] = None
                 ) -> None:
        super().__init__()
        if norm is None:
            norm = nn.BatchNorm2d

        width = int(channels * (base_width / 64.0)) * groups

        self.bn1 = norm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, stride=1)

        self.bn2 = norm(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, pad_mode='pad', group=groups)

        self.bn3 = norm(width)
        self.conv3 = nn.Conv2d(width, channels * self.expansion,
                               kernel_size=1, stride=1)

        self.relu = nn.ReLU()
        self.down_sample = down_sample

    def construct(self, x: Tensor) -> Tensor:
        identity = x

        out = self.bn1(x)
        out = self.relu(out)

        residual = out

        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.down_sample is not None:
            identity = self.down_sample(residual)

        out += identity

        return out


@register_model
def resnetv2_50(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    """Get 50 layers ResNetV2 model.
    Refer to the base class `models.ResNet` for more details.
    """
    default_cfg = default_cfgs['resnetv2_50']
    model = ResNet(PreActBottleneck, [3, 4, 6, 3], num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def resnetv2_101(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    """Get 101 layers ResNetV2 model.
    Refer to the base class `models.ResNet` for more details.
    """
    default_cfg = default_cfgs["resnetv2_101"]
    model = ResNet(PreActBottleneck, [3, 4, 23, 3], num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model
