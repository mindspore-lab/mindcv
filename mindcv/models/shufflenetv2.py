"""
MindSpore implementation of `ShuffleNetV2`.
Refer to ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design
"""

from typing import Tuple

import mindspore.common.initializer as init
from mindspore import Tensor, mint, nn

from .helpers import load_pretrained
from .layers.pooling import GlobalAvgPooling
from .registry import register_model

__all__ = [
    "ShuffleNetV2",
    "shufflenet_v2_x0_5",
    "shufflenet_v2_x1_0",
    "shufflenet_v2_x1_5",
    "shufflenet_v2_x2_0",
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
    "shufflenet_v2_x0_5": _cfg(
        url="https://download.mindspore.cn/toolkits/mindcv/shufflenet/shufflenetv2/shufflenet_v2_x0_5-8c841061.ckpt"
    ),
    "shufflenet_v2_x1_0": _cfg(
        url="https://download.mindspore.cn/toolkits/mindcv/shufflenet/shufflenetv2/shufflenet_v2_x1_0-0da4b7fa.ckpt"
    ),
    "shufflenet_v2_x1_5": _cfg(
        url="https://download.mindspore.cn/toolkits/mindcv/shufflenet/shufflenetv2/shufflenet_v2_x1_5-00b56131.ckpt"
    ),
    "shufflenet_v2_x2_0": _cfg(
        url="https://download.mindspore.cn/toolkits/mindcv/shufflenet/shufflenetv2/shufflenet_v2_x2_0-ed8e698d.ckpt"
    ),
}


class ShuffleV2Block(nn.Cell):
    """define the basic block of ShuffleV2"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int,
        kernel_size: int,
        stride: int,
    ) -> None:
        super().__init__()
        assert stride in [1, 2]
        self.stride = stride
        pad = kernel_size // 2
        out_channels = out_channels - in_channels
        branch_main = [
            # pw
            mint.nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, bias=False),
            mint.nn.BatchNorm2d(mid_channels),
            mint.nn.ReLU(),
            # dw
            mint.nn.Conv2d(
                mid_channels, mid_channels, kernel_size=kernel_size, stride=stride,
                padding=pad, groups=mid_channels, bias=False
            ),
            mint.nn.BatchNorm2d(mid_channels),
            # pw-linear
            mint.nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, bias=False),
            mint.nn.BatchNorm2d(out_channels),
            mint.nn.ReLU(),
        ]
        self.branch_main = nn.SequentialCell(branch_main)

        if stride == 2:
            branch_proj = [
                # dw
                mint.nn.Conv2d(
                    in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                    padding=pad, groups=in_channels, bias=False
                ),
                mint.nn.BatchNorm2d(in_channels),
                # pw-linear
                mint.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, bias=False),
                mint.nn.BatchNorm2d(in_channels),
                mint.nn.ReLU(),
            ]
            self.branch_proj = nn.SequentialCell(branch_proj)
        else:
            self.branch_proj = None

    def construct(self, old_x: Tensor) -> Tensor:
        if self.stride == 1:
            x_proj, x = self.channel_shuffle(old_x)
            return mint.concat((x_proj, self.branch_main(x)), dim=1)

        if self.stride == 2:
            x_proj = old_x
            x = old_x
            return mint.concat((self.branch_proj(x_proj), self.branch_main(x)), dim=1)
        return None

    @staticmethod
    def channel_shuffle(x: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size, num_channels, height, width = x.shape
        x = mint.reshape(x, (batch_size * num_channels // 2, 2, height * width,))
        x = mint.permute(x, (1, 0, 2,))
        x = mint.reshape(x, (2, -1, num_channels // 2, height, width,))
        return x[0], x[1]


class ShuffleNetV2(nn.Cell):
    r"""ShuffleNetV2 model class, based on
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design" <https://arxiv.org/abs/1807.11164>`_

    Args:
        num_classes: number of classification classes. Default: 1000.
        in_channels: number of input channels. Default: 3.
        model_size: scale factor which controls the number of channels. Default: '1.5x'.
    """

    def __init__(
        self,
        num_classes: int = 1000,
        in_channels: int = 3,
        model_size: str = "1.5x",
    ):
        super().__init__()

        self.stage_repeats = [4, 8, 4]
        self.model_size = model_size
        if model_size == "0.5x":
            self.stage_out_channels = [-1, 24, 48, 96, 192, 1024]
        elif model_size == "1.0x":
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif model_size == "1.5x":
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif model_size == "2.0x":
            self.stage_out_channels = [-1, 24, 244, 488, 976, 2048]
        else:
            raise NotImplementedError

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.first_conv = nn.SequentialCell([
            mint.nn.Conv2d(in_channels, input_channel, kernel_size=3, stride=2, padding=1, bias=False),
            mint.nn.BatchNorm2d(input_channel),
            mint.nn.ReLU(),
        ])
        self.max_pool = mint.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.features = []
        for idxstage, numrepeat in enumerate(self.stage_repeats):
            output_channel = self.stage_out_channels[idxstage + 2]
            for i in range(numrepeat):
                if i == 0:
                    self.features.append(ShuffleV2Block(input_channel, output_channel,
                                                        mid_channels=output_channel // 2, kernel_size=3, stride=2))
                else:
                    self.features.append(ShuffleV2Block(input_channel // 2, output_channel,
                                                        mid_channels=output_channel // 2, kernel_size=3, stride=1))
                input_channel = output_channel

        self.features = nn.SequentialCell(self.features)

        self.conv_last = nn.SequentialCell([
            mint.nn.Conv2d(input_channel, self.stage_out_channels[-1], kernel_size=1, stride=1, bias=False),
            mint.nn.BatchNorm2d(self.stage_out_channels[-1]),
            mint.nn.ReLU()
        ])
        self.pool = GlobalAvgPooling()
        self.classifier = mint.nn.Linear(self.stage_out_channels[-1], num_classes, bias=False)
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for cells."""
        for name, cell in self.cells_and_names():
            if isinstance(cell, mint.nn.Conv2d):
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
            elif isinstance(cell, mint.nn.Linear):
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
        x = self.conv_last(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x

    def construct(self, x: Tensor) -> Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


@register_model
def shufflenet_v2_x0_5(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> ShuffleNetV2:
    """Get ShuffleNetV2 model with width scaled by 0.5.
    Refer to the base class `models.ShuffleNetV2` for more details.
    """
    default_cfg = default_cfgs["shufflenet_v2_x0_5"]
    model = ShuffleNetV2(model_size="0.5x", num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def shufflenet_v2_x1_0(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> ShuffleNetV2:
    """Get ShuffleNetV2 model with width scaled by 1.0.
    Refer to the base class `models.ShuffleNetV2` for more details.
    """
    default_cfg = default_cfgs["shufflenet_v2_x1_0"]
    model = ShuffleNetV2(model_size="1.0x", num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def shufflenet_v2_x1_5(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> ShuffleNetV2:
    """Get ShuffleNetV2 model with width scaled by 1.5.
    Refer to the base class `models.ShuffleNetV2` for more details.
    """
    default_cfg = default_cfgs["shufflenet_v2_x1_5"]
    model = ShuffleNetV2(model_size="1.5x", num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def shufflenet_v2_x2_0(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> ShuffleNetV2:
    """Get ShuffleNetV2 model with width scaled by 2.0.
    Refer to the base class `models.ShuffleNetV2` for more details.
    """
    default_cfg = default_cfgs["shufflenet_v2_x2_0"]
    model = ShuffleNetV2(model_size="2.0x", num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model
