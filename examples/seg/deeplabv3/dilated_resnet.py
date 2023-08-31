"""mindcv resnet with atrous convolution applied"""

from typing import List, Optional, Type
from mindspore import nn
from mindcv.models.resnet import Bottleneck
from mindcv.models.resnet import ResNet, Bottleneck
from mindcv.models.helpers import build_model_with_cfg
from mindcv.models.registry import register_model


__all__ = [
    "DilatedResNet",
    "dilated_resnet50",
    "dilated_resnet101",
]

def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "first_conv": "conv1",
        "classifier": "classifier",
        **kwargs,
    }

default_cfgs = {
    "resnet50": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/resnet/resnet50-e0733ab8.ckpt"),
    "resnet101": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/resnet/resnet101-689c5e77.ckpt"),
}    

def conv1x1(in_channels, out_channels, stride=1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(
        in_channels, 
        out_channels, 
        kernel_size=1, 
        stride=stride)


def conv3x3(in_channels, out_channels, stride=1, dilation=1, padding=1) -> nn.Conv2d :
    """3x3 convolution"""
    return nn.Conv2d(
        in_channels, 
        out_channels, 
        kernel_size=3, 
        stride=stride, 
        pad_mode='pad', 
        padding=padding,
        dilation=dilation)

class DilatedBottleneck(Bottleneck):
    def __init__(
        self, 
        in_channels: int, 
        channels: int, 
        stride: int = 1, 
        groups: int = 1, 
        dilation: int = 1,
        base_width: int = 64, 
        norm: Optional[nn.Cell] = None,
        down_sample: Optional[nn.Cell] = None,
    ) -> None:
        
        nn.Cell.__init__(self)
        if norm is None:
            norm = nn.BatchNorm2d

        width = int(channels * (base_width / 64.0)) * groups

        self.conv1 = self.conv1 = conv1x1(in_channels, channels)
        self.bn1 = norm(width)
        self.conv2 = conv3x3(channels, channels, stride=stride, 
                             dilation=dilation, padding=dilation)
        self.bn2 = norm(width)
        self.conv3 = conv1x1(width, channels * self.expansion)
        self.bn3 = norm(channels * self.expansion)
        self.relu = nn.ReLU()
        self.down_sample = down_sample        


class DilatedResNet(ResNet):
    def __init__(
        self, 
        block: Type[DilatedBottleneck], 
        layers: List[int], 
        output_stride: int = 16,
        in_channels: int = 3, 
        groups: int = 1, 
        base_width: int = 64, 
        norm: Optional[nn.Cell] = None,
    ) -> None:
        super().__init__(block=block, layers=layers)
        if norm is None:
            norm = nn.BatchNorm2d
        
        self.norm: nn.Cell = norm  
        self.input_channels = 64
        self.groups = groups
        self.base_with = base_width

        self.conv1 = nn.Conv2d(in_channels, self.input_channels, kernel_size=7,
                               stride=2, pad_mode="pad", padding=3)
        self.bn1 = norm(self.input_channels)
        self.relu = nn.ReLU()
        self.feature_info = [dict(chs=self.input_channels, reduction=2, name="relu")]

        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.feature_info.append(dict(chs=block.expansion * 64, reduction=4, name="layer1"))

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.feature_info.append(dict(chs=block.expansion * 128, reduction=8, name="layer2"))

        if output_stride == 16:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
            self.feature_info.append(dict(chs=block.expansion * 256, reduction=output_stride, name="layer3"))
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1, base_dilation=2, grids=[1, 2, 4])
            self.feature_info.append(dict(chs=block.expansion * 512, reduction=output_stride, name="layer4"))
        
        elif output_stride == 8:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1, base_dilation=2)
            self.feature_info.append(dict(chs=block.expansion * 256, reduction=output_stride, name="layer3"))
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1, base_dilation=4, grids=[1, 2, 4])
            self.feature_info.append(dict(chs=block.expansion * 512, reduction=output_stride, name="layer4"))

        else:
            raise ValueError('output_stride {} not supported'.format(output_stride))
        
                     
    def _make_layer(
        self,
        block: Type[DilatedBottleneck],
        channels: int,
        block_nums: int,
        stride: int = 1,
        base_dilation: int = 1,
        grids: list = None,
    ) -> nn.SequentialCell:
        """build model depending on cfgs"""
        down_sample = None

        if stride != 1 or self.input_channels != channels * block.expansion:
            down_sample = nn.SequentialCell([
                conv1x1(self.input_channels, channels * block.expansion, stride),
                self.norm(channels * block.expansion)
            ])

        if grids is None:
            grids = [1] * block_nums

        layers = []
        layers.append(
            block(
                self.input_channels,
                channels,
                stride=stride,
                down_sample=down_sample,
                groups=self.groups,
                base_width=self.base_with,
                norm=self.norm,
                dilation = base_dilation * grids[0],
            )
        )

        self.input_channels = channels * block.expansion
        for i in range(1, block_nums):
            layers.append(
                block(
                    self.input_channels,
                    channels,
                    groups=self.groups,
                    base_width=self.base_with,
                    norm=self.norm,
                    dilation = base_dilation * grids[i],
                )
            )

        return nn.SequentialCell(layers)
   
def _create_resnet(pretrained=False, **kwargs):
    return build_model_with_cfg(DilatedResNet, pretrained, **kwargs)

@register_model
def dilated_resnet50(pretrained: bool = False, num_classes: int = 1000, in_channels: int = 3, 
                     output_stride:int = 16, **kwargs):
    """Get 101 layers ResNet model with dilation.
    Refer to the base class `DilatedResNet` for more details.
    """
    default_cfg = default_cfgs["resnet50"]
    model_args = dict(block=DilatedBottleneck, layers=[3, 4, 6, 3], num_classes=num_classes, 
                      in_channels=in_channels, output_stride=output_stride,
                      **kwargs)
    return _create_resnet(pretrained, **dict(default_cfg=default_cfg, **model_args))


@register_model
def dilated_resnet101(pretrained: bool = False, num_classes: int = 1000, in_channels=3, 
                      output_stride:int = 16, **kwargs):
    """Get 101 layers ResNet model with dilation.
    Refer to the base class `DilatedResNet` for more details.
    """
    default_cfg = default_cfgs["resnet101"]
    model_args = dict(block=DilatedBottleneck, layers=[3, 4, 23, 3], num_classes=num_classes, 
                      in_channels=in_channels, output_stride=output_stride,
                      **kwargs)
    return _create_resnet(pretrained, **dict(default_cfg=default_cfg, **model_args))