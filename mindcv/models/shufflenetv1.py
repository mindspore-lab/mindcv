"""
MindSpore implementation of `ShuffleNetV1`.
Refer to ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices
"""

import mindspore.common.initializer as init
from mindspore import Tensor, nn, ops

from .helpers import load_pretrained
from .layers.compatibility import Split
from .layers.pooling import GlobalAvgPooling
from .registry import register_model

__all__ = [
    "ShuffleNetV1",
    "shufflenet_v1_g3_05",
    "shufflenet_v1_g3_10",
    "shufflenet_v1_g3_15",
    "shufflenet_v1_g3_20",
    "shufflenet_v1_g8_05",
    "shufflenet_v1_g8_10",
    "shufflenet_v1_g8_15",
    "shufflenet_v1_g8_20",
]


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "first_conv": "first_conv.0",
        "classifier": "classifier",
        **kwargs,
    }


default_cfgs = {
    "shufflenet_v1_g3_05": _cfg(
        url="https://download.mindspore.cn/toolkits/mindcv/shufflenet/shufflenetv1/shufflenet_v1_g3_05-42cfe109.ckpt"
    ),
    "shufflenet_v1_g3_10": _cfg(
        url="https://download.mindspore.cn/toolkits/mindcv/shufflenet/shufflenetv1/shufflenet_v1_g3_10-245f0ccf.ckpt"
    ),
    "shufflenet_v1_g3_15": _cfg(url=""),
    "shufflenet_v1_g3_20": _cfg(url=""),
    "shufflenet_v1_g8_05": _cfg(url=""),
    "shufflenet_v1_g8_10": _cfg(url=""),
    "shufflenet_v1_g8_15": _cfg(url=""),
    "shufflenet_v1_g8_20": _cfg(url=""),
}


class GroupConv(nn.Cell):
    """
    Group convolution operation.
    Due to MindSpore doesn't support group conv in shufflenet, we need to define the group convolution manually, instead
    of using the origin nn.Conv2d by changing the parameter `group`.

    Args:
        in_channels (int): Input channels of feature map.
        out_channels (int): Output channels of feature map.
        kernel_size (int): Size of convolution kernel.
        stride (int): Stride size for the group convolution layer.
        pad_mode (str): Specifies padding mode.
        pad (int): The number of padding on the height and width directions of the input.
        groups (int): Splits filter into groups, `in_channels` and `out_channels` must be divisible by `group`.
        has_bias (bool): Whether the Conv2d layer has a bias parameter.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, pad_mode="pad", pad=0, groups=1, has_bias=False):
        super(GroupConv, self).__init__()
        assert in_channels % groups == 0 and out_channels % groups == 0
        self.groups = groups
        self.convs = nn.CellList()
        self.split = Split(split_size_or_sections=in_channels // groups, output_num=self.groups, axis=1)
        for _ in range(groups):
            self.convs.append(
                nn.Conv2d(
                    in_channels // groups,
                    out_channels // groups,
                    kernel_size=kernel_size,
                    stride=stride,
                    has_bias=has_bias,
                    padding=pad,
                    pad_mode=pad_mode,
                )
            )

    def construct(self, x):
        features = self.split(x)
        outputs = ()
        for i in range(self.groups):
            outputs = outputs + (self.convs[i](features[i]),)
        out = ops.concat(outputs, axis=1)
        return out


class ShuffleV1Block(nn.Cell):
    """Basic block of ShuffleNetV1. 1x1 GC -> CS -> 3x3 DWC -> 1x1 GC"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int,
        stride: int,
        group: int,
        first_group: bool,
    ) -> None:
        super().__init__()
        assert stride in [1, 2]
        self.stride = stride
        self.group = group

        if stride == 2:
            out_channels = out_channels - in_channels

        branch_main_1 = [
            # pw
            GroupConv(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=1,
                stride=1,
                pad_mode="pad",
                pad=0,
                groups=1 if first_group else group,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
        ]

        branch_main_2 = [
            # dw
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, pad_mode="pad", padding=1,
                      group=mid_channels),
            nn.BatchNorm2d(mid_channels),
            # pw-linear
            GroupConv(
                in_channels=mid_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                pad_mode="pad",
                pad=0,
                groups=group,
            ),
            nn.BatchNorm2d(out_channels),
        ]
        self.branch_main_1 = nn.SequentialCell(branch_main_1)
        self.branch_main_2 = nn.SequentialCell(branch_main_2)
        if stride == 2:
            self.branch_proj = nn.AvgPool2d(kernel_size=3, stride=2, pad_mode="same")

        self.relu = nn.ReLU()

    def construct(self, x: Tensor) -> Tensor:
        identify = x
        x = self.branch_main_1(x)
        if self.group > 1:
            x = self.channel_shuffle(x)
        x = self.branch_main_2(x)
        if self.stride == 1:
            out = self.relu(identify + x)
        else:
            out = self.relu(ops.concat((self.branch_proj(identify), x), axis=1))

        return out

    def channel_shuffle(self, x: Tensor) -> Tensor:
        batch_size, num_channels, height, width = x.shape

        group_channels = num_channels // self.group
        x = ops.reshape(x, (batch_size, group_channels, self.group, height, width))
        x = ops.transpose(x, (0, 2, 1, 3, 4))
        x = ops.reshape(x, (batch_size, num_channels, height, width))
        return x


class ShuffleNetV1(nn.Cell):
    r"""ShuffleNetV1 model class, based on
    `"ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices" <https://arxiv.org/abs/1707.01083>`_  # noqa: E501

    Args:
        num_classes: number of classification classes. Default: 1000.
        in_channels: number of input channels. Default: 3.
        model_size: scale factor which controls the number of channels. Default: '2.0x'.
        group: number of group for group convolution. Default: 3.
    """

    def __init__(
        self,
        num_classes: int = 1000,
        in_channels: int = 3,
        model_size: str = "2.0x",
        group: int = 3,
    ):
        super().__init__()
        self.stage_repeats = [4, 8, 4]
        self.model_size = model_size
        if group == 3:
            if model_size == "0.5x":
                self.stage_out_channels = [-1, 12, 120, 240, 480]
            elif model_size == "1.0x":
                self.stage_out_channels = [-1, 24, 240, 480, 960]
            elif model_size == "1.5x":
                self.stage_out_channels = [-1, 24, 360, 720, 1440]
            elif model_size == "2.0x":
                self.stage_out_channels = [-1, 48, 480, 960, 1920]
            else:
                raise NotImplementedError
        elif group == 8:
            if model_size == "0.5x":
                self.stage_out_channels = [-1, 16, 192, 384, 768]
            elif model_size == "1.0x":
                self.stage_out_channels = [-1, 24, 384, 768, 1536]
            elif model_size == "1.5x":
                self.stage_out_channels = [-1, 24, 576, 1152, 2304]
            elif model_size == "2.0x":
                self.stage_out_channels = [-1, 48, 768, 1536, 3072]
            else:
                raise NotImplementedError

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.first_conv = nn.SequentialCell(
            nn.Conv2d(in_channels, input_channel, kernel_size=3, stride=2, pad_mode="pad", padding=1),
            nn.BatchNorm2d(input_channel),
            nn.ReLU(),
        )
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")

        features = []
        for idxstage, numrepeat in enumerate(self.stage_repeats):
            output_channel = self.stage_out_channels[idxstage + 2]
            for i in range(numrepeat):
                stride = 2 if i == 0 else 1
                first_group = idxstage == 0 and i == 0
                features.append(ShuffleV1Block(input_channel, output_channel,
                                               group=group, first_group=first_group,
                                               mid_channels=output_channel // 4, stride=stride))
                input_channel = output_channel

        self.features = nn.SequentialCell(features)
        self.global_pool = GlobalAvgPooling()
        self.classifier = nn.Dense(self.stage_out_channels[-1], num_classes, has_bias=False)
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for cells."""
        for name, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                if "first" in name:
                    cell.weight.set_data(
                        init.initializer(init.Normal(0.01, 0), cell.weight.shape, cell.weight.dtype))
                else:
                    cell.weight.set_data(
                        init.initializer(init.Normal(1.0 / cell.weight.shape[1], 0), cell.weight.shape,
                                         cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(
                        init.initializer("zeros", cell.bias.shape, cell.bias.dtype))
            elif isinstance(cell, nn.Dense):
                cell.weight.set_data(
                    init.initializer(init.Normal(0.01, 0), cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(
                        init.initializer("zeros", cell.bias.shape, cell.bias.dtype))

    def forward_features(self, x: Tensor) -> Tensor:
        x = self.first_conv(x)
        x = self.max_pool(x)
        x = self.features(x)
        return x

    def forward_head(self, x: Tensor) -> Tensor:
        x = self.global_pool(x)
        x = self.classifier(x)
        return x

    def construct(self, x: Tensor) -> Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


@register_model
def shufflenet_v1_g3_05(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> ShuffleNetV1:
    """Get ShuffleNetV1 model with width scaled by 0.5 and 3 groups of GPConv.
    Refer to the base class `models.ShuffleNetV1` for more details.
    """
    default_cfg = default_cfgs["shufflenet_v1_g3_05"]
    model = ShuffleNetV1(group=3, model_size="0.5x", num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def shufflenet_v1_g3_10(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> ShuffleNetV1:
    """Get ShuffleNetV1 model with width scaled by 1.0 and 3 groups of GPConv.
    Refer to the base class `models.ShuffleNetV1` for more details.
    """
    default_cfg = default_cfgs["shufflenet_v1_g3_10"]
    model = ShuffleNetV1(group=3, model_size="1.0x", num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def shufflenet_v1_g3_15(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> ShuffleNetV1:
    """Get ShuffleNetV1 model with width scaled by 1.5 and 3 groups of GPConv.
    Refer to the base class `models.ShuffleNetV1` for more details.
    """
    default_cfg = default_cfgs["shufflenet_v1_g3_15"]
    model = ShuffleNetV1(group=3, model_size="1.5x", num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def shufflenet_v1_g3_20(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> ShuffleNetV1:
    """Get ShuffleNetV1 model with width scaled by 2.0 and 3 groups of GPConv.
    Refer to the base class `models.ShuffleNetV1` for more details.
    """
    default_cfg = default_cfgs["shufflenet_v1_g3_20"]
    model = ShuffleNetV1(group=3, model_size="2.0x", num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def shufflenet_v1_g8_05(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> ShuffleNetV1:
    """Get ShuffleNetV1 model with width scaled by 0.5 and 8 groups of GPConv.
    Refer to the base class `models.ShuffleNetV1` for more details.
    """
    default_cfg = default_cfgs["shufflenet_v1_g8_05"]
    model = ShuffleNetV1(group=8, model_size="0.5x", num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def shufflenet_v1_g8_10(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> ShuffleNetV1:
    """Get ShuffleNetV1 model with width scaled by 1.0 and 8 groups of GPConv.
    Refer to the base class `models.ShuffleNetV1` for more details.
    """
    default_cfg = default_cfgs["shufflenet_v1_g8_10"]
    model = ShuffleNetV1(group=8, model_size="1.0x", num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def shufflenet_v1_g8_15(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> ShuffleNetV1:
    """Get ShuffleNetV1 model with width scaled by 1.5 and 8 groups of GPConv.
    Refer to the base class `models.ShuffleNetV1` for more details.
    """
    default_cfg = default_cfgs["shufflenet_v1_g8_15"]
    model = ShuffleNetV1(group=8, model_size="1.5x", num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def shufflenet_v1_g8_20(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> ShuffleNetV1:
    """Get ShuffleNetV1 model with width scaled by 2.0 and 8 groups of GPConv.
    Refer to the base class `models.ShuffleNetV1` for more details.
    """
    default_cfg = default_cfgs["shufflenet_v1_g8_20"]
    model = ShuffleNetV1(group=8, model_size="2.0x", num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model
