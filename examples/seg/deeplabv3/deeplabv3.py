"""DeeplabV3, DeeplabV3+ implement with replaceable backbones"""

from typing import List

import mindspore.nn as nn
import mindspore.ops as ops


def conv1x1(in_channels, out_channels, stride=1, weight_init="xavier_uniform") -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, weight_init=weight_init)


def conv3x3(in_channels, out_channels, stride=1, dilation=1, padding=1, weight_init="xavier_uniform") -> nn.Conv2d:
    """3x3 convolution"""
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        pad_mode="pad",
        padding=padding,
        dilation=dilation,
        weight_init=weight_init,
    )


class ASPP(nn.Cell):
    """
    Atrous Spatial Pyramid Pooling.
    """

    def __init__(
        self,
        atrous_rates: List[int],
        is_training: bool = True,
        in_channels: int = 2048,
        out_channels: int = 256,
        num_classes: int = 21,
        weight_init: str = "xavier_uniform",
    ) -> None:
        super(ASPP, self).__init__()

        self.is_training = is_training

        self.aspp_convs = nn.CellList()
        for rate in atrous_rates:
            self.aspp_convs.append(
                ASPPConv(
                    in_channels,
                    out_channels,
                    rate,
                )
            )
        self.aspp_convs.append(ASPPPooling(in_channels, out_channels))

        self.conv1 = nn.Conv2d(
            out_channels * (len(atrous_rates) + 1), out_channels, kernel_size=1, weight_init=weight_init
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.7)
        # self.conv2 = nn.Conv2d(out_channels, num_classes, kernel_size=1, weight_init=weight_init, has_bias=True)

    def construct(self, x):
        _out = []
        for conv in self.aspp_convs:
            _out.append(conv(x))
        x = ops.cat(_out, axis=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.is_training:
            x = self.drop(x)
        # x = self.conv2(x)
        return x


class ASPPPooling(nn.Cell):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        weight_init: str = "xavier_uniform",
    ) -> None:
        super(ASPPPooling, self).__init__()
        self.conv = nn.SequentialCell(
            [
                nn.Conv2d(in_channels, out_channels, kernel_size=1, weight_init=weight_init),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            ]
        )

    def construct(self, x):
        size = ops.shape(x)
        out = ops.mean(x, (2, 3), True)
        out = self.conv(out)
        out = ops.ResizeNearestNeighbor((size[2], size[3]), True)(out)
        return out


class ASPPConv(nn.Cell):
    def __init__(
        self, in_channels: int, out_channels: int, atrous_rate: int = 1, weight_init: str = "xavier_uniform"
    ) -> None:
        super(ASPPConv, self).__init__()

        self._aspp_conv = nn.SequentialCell(
            [
                nn.Conv2d(in_channels, out_channels, kernel_size=1, has_bias=False, weight_init=weight_init)
                if atrous_rate == 1
                else nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    pad_mode="pad",
                    padding=atrous_rate,
                    dilation=atrous_rate,
                    weight_init=weight_init,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            ]
        )

    def construct(self, x):
        out = self._aspp_conv(x)
        return out


class DeepLabV3(nn.Cell):
    """
    Constructs a DeepLabV3 model with input backbone.
    Reference: `Rethinking Atrous Convolution for Semantic Image Segmentation <https://arxiv.org/abs/1706.05587>`__.  # noqa: E501

    Args:
        backbone (Cell): Backbone Network.
        args (dict): The default config of DeepLabV3.
        is_training (bool): default True.
    """

    def __init__(
        self,
        backbone,
        args,
        is_training: bool = True,
    ):
        super(DeepLabV3, self).__init__()
        self.is_training = is_training
        self.backbone = backbone
        self.aspp = ASPP(
            atrous_rates=[1, 6, 12, 18],
            is_training=is_training,
            in_channels=2048,
            num_classes=args.num_classes,
        )
        self.conv = nn.Conv2d(256, args.num_classes, kernel_size=1, weight_init="xavier_uniform", has_bias=True)

    def construct(self, x):
        size = ops.shape(x)
        features = self.backbone(x)[-1]
        out = self.aspp(features)
        out = self.conv(out)
        out = ops.interpolate(out, size=(size[2], size[3]), mode="bilinear", align_corners=True)
        return out


class Decoder(nn.Cell):
    """
    Decoder module of DeepLabV3+ model.
    """

    def __init__(
        self,
        low_level_channels: int = 256,
        num_classes: int = 21,
        weight_init: str = "xavier_uniform",
    ):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(low_level_channels, 48, kernel_size=1, weight_init=weight_init)
        self.bn1 = nn.BatchNorm2d(48)
        self.relu = nn.ReLU()

        self.last_conv = nn.SequentialCell(
            [
                conv3x3(304, 256, stride=1, dilation=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                conv3x3(256, 256, stride=1, dilation=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                conv1x1(256, num_classes, stride=1),
            ]
        )

    def construct(self, x, low_level_features):
        size = ops.shape(low_level_features)
        low_level_features = self.conv1(low_level_features)
        low_level_features = self.bn1(low_level_features)
        low_level_features = self.relu(low_level_features)

        x = ops.ResizeNearestNeighbor(size=(size[2], size[3]), align_corners=True)(x)
        x = ops.cat((x, low_level_features), axis=1)
        x = self.last_conv(x)

        return x


class DeepLabV3Plus(nn.Cell):
    """
    Constructs a DeepLabV3+ model with input backbone.
    Reference: `Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation <https://arxiv.org/abs/1802.02611>`__.  # noqa: E501

    Args:
        backbone (Cell): Backbone Network.
        args (dict): The default config of DeepLabV3+.
        is_training (bool): default True.
    """

    def __init__(
        self,
        backbone,
        args,
        is_training: bool = True,
    ):
        super(DeepLabV3Plus, self).__init__()
        self.is_training = is_training
        self.backbone = backbone
        self.aspp = ASPP(
            atrous_rates=[1, 6, 12, 18],
            is_training=is_training,
            in_channels=2048,
            num_classes=args.num_classes,
        )
        # low_level_channels : Resnet-256, Xception-128
        self.decoder = Decoder(low_level_channels=256, num_classes=args.num_classes)

    def construct(self, x):
        size = ops.shape(x)
        low_level_features, features = self.backbone(x)
        out = self.aspp(features)
        out = self.decoder(out, low_level_features)
        out = ops.interpolate(out, size=(size[2], size[3]), mode="bilinear", align_corners=True)
        return out


class DeepLabInferNetwork(nn.Cell):
    """
    Provide infer network of Deeplab, network could be deeplabv3 or deeplabv3+.

    Args:
        network (Cell): DeepLabV3 or DeeplabV3Plus.
        input_format (str): format of input data, "NCHW" or "NHWC".

    Example:
        deeplab = DeepLabV3(backbone, args)
        eval_model = DeeplabInferNetwork(deeplab, input_format="NCHW")
    """

    def __init__(self, network, input_format="NCHW"):
        super(DeepLabInferNetwork, self).__init__(auto_prefix=False)
        self.network = network
        self.softmax = nn.Softmax(axis=1)
        self.format = input_format

    def construct(self, input_data):
        if self.format == "NHWC":
            input_data = ops.transpose(input_data, (0, 3, 1, 2))
        output = self.network(input_data)
        output = self.softmax(output)
        return output
