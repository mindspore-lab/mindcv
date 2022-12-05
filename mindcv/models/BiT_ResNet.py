"""
MindSpore implementation of `ResNet`.
Refer to Deep Residual Learning for Image Recognition.
"""

from typing import Optional, Type, List, Union

import mindspore
from mindspore import nn, Tensor, ops



from .layers.pooling import GlobalAvgPooling
from .utils import load_pretrained
from .registry import register_model

__all__ = [
    'ResNet',
    'BiTresnet50',
]


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000,
        'first_conv': 'conv1', 'classifier': 'classifier',
        **kwargs
    }


class StdConv2d(nn.Conv2d):

    def construct(self, x):
        w = self.weight
        mean_op = ops.ReduceMean(keep_dims=True)
        m = mean_op(w, [1, 2, 3])
        v = w.var((1, 2, 3), keepdims=True)
        w = (w - m) / mindspore.ops.sqrt(v + 1e-10)
        output = self.conv2d(x, w)
        return output

class Bottleneck(nn.Cell):
    """
    Bottleneck here places the stride for downsampling at 3x3 convolution(self.conv2) as torchvision does,
    while original implementation places the stride at the first 1x1 convolution(self.conv1)
    """
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
            norm = nn.GroupNorm

        width = int(channels * (base_width / 64.0)) * groups
        self.gn1 = norm(32,in_channels)
        self.conv1 = StdConv2d(in_channels, width, kernel_size=1, stride=1)
        self.gn2 = norm(32,width)
        self.conv2 = StdConv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, pad_mode='pad', group=groups)
        self.gn3 = norm(32,width)
        self.conv3 = StdConv2d(width, channels * self.expansion,
                               kernel_size=1, stride=1)

        self.relu = nn.ReLU()
        self.down_sample = down_sample

    def construct(self, x: Tensor) -> Tensor:
        identity = x
        out = self.gn1(x)
        out = self.relu(out)

        residual = out

        out = self.conv1(out)

        out = self.gn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.gn3(out)
        out = self.relu(out)
        out = self.conv3(out)


        if self.down_sample is not None:
            identity = self.down_sample(residual)

        out += identity
        # out = self.relu(out)

        return out


class ResNet(nn.Cell):
    r"""ResNet model class, based on
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/abs/1512.03385>`_

    Args:
        block: block of resnet.
        layers: number of layers of each stage.
        num_classes: number of classification classes. Default: 1000.
        in_channels: number the channels of the input. Default: 3.
        groups: number of groups for group conv in blocks. Default: 1.
        base_width: base width of pre group hidden channel in blocks. Default: 64.
        norm: normalization layer in blocks. Default: None.
    """

    def __init__(self,
                 block: Type[Union[Bottleneck]],
                 layers: List[int],
                 wf: int = 1,
                 num_classes: int = 1000,
                 in_channels: int = 3,
                 groups: int = 1,
                 base_width: int = 64,
                 norm: Optional[nn.Cell] = None
                 ) -> None:
        super().__init__()

        if norm is None:
            norm = nn.GroupNorm

        self.norm: nn.Cell = norm  # add type hints to make pylint happy
        self.input_channels = 64*wf
        self.groups = groups
        self.base_with = base_width

        #root
        self.conv1 = StdConv2d(in_channels, self.input_channels, kernel_size=7,
                               stride=2, pad_mode='pad', padding=3)
        self.pad = nn.ConstantPad2d(1,0)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='valid')

        #body
        self.layer1 = self._make_layer(block, 64*wf, layers[0])
        self.layer2 = self._make_layer(block, 128*wf, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256*wf, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512*wf, layers[3], stride=2)

        #head
        self.gn = norm(32,2048*wf)
        self.relu = nn.ReLU()
        self.pool = GlobalAvgPooling(keep_dims=True)
        self.classifier = nn.Conv2d(512 * block.expansion * wf,num_classes,kernel_size=1,has_bias=True)
    def _make_layer(self,
                    block: Type[Union[Bottleneck]],
                    channels: int,
                    block_nums: int,
                    stride: int = 1
                    ) -> nn.SequentialCell:
        """build model depending on cfgs"""
        down_sample = None

        if stride != 1 or self.input_channels != channels * block.expansion:
            down_sample = nn.SequentialCell([
                StdConv2d(self.input_channels, channels * block.expansion, kernel_size=1, stride=stride),
            ])

        layers = []
        layers.append(
            block(
                self.input_channels,
                channels,
                stride=stride,
                down_sample=down_sample,
                groups=self.groups,
                base_width=self.base_with,
                norm=self.norm
            )
        )
        self.input_channels = channels * block.expansion

        for _ in range(1, block_nums):
            layers.append(
                block(
                    self.input_channels,
                    channels,
                    groups=self.groups,
                    base_width=self.base_with,
                    norm=self.norm
                )
            )

        return nn.SequentialCell(layers)

    def root(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.pad(x)
        x = self.max_pool(x)
        return x

    def forward_features(self, x: Tensor) -> Tensor:
        """Network forward feature extraction."""
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward_head(self, x: Tensor) -> Tensor:
        x = self.gn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x

    def construct(self, x: Tensor) -> Tensor:
        x = self.root(x)
        x = self.forward_features(x)
        x = self.forward_head(x)
        assert x.shape[-2:] == (1, 1)  # We should have no spatial shape left.
        return x[...,0,0]

@register_model
def BiTresnet50(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    """Get 50 layers ResNet model.
     Refer to the base class `models.ResNet` for more details.
     """
    default_cfg = default_cfgs['resnet50']
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model

