"""DeeplabV3 implement with replaceable backbones"""

from typing import List

import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor


class ASPP(nn.Cell):
    """
    Atrous Spatial Pyramid Pooling.

    """

    def __init__(
        self,
        atrous_rates: List[int],
        is_training: bool = True,
        # phase='train',
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
        self.conv2 = nn.Conv2d(out_channels, num_classes, kernel_size=1, weight_init=weight_init, has_bias=True)

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
        x = self.conv2(x)
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
        out = nn.AvgPool2d(size[2])(x)
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
    DeeplabV3 implement.

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

    def construct(self, x):
        size = ops.shape(x)
        features = self.backbone(x)[-1]
        out = self.aspp(features)
        out = ops.interpolate(out, size=(size[2], size[3]), mode="bilinear", align_corners=True)
        return out


class DeepLabV3InferNetwork(nn.Cell):
    """
    Provide DeeplabV3 infer network.

    """

    def __init__(self, network, input_format="NCHW"):
        super(DeepLabV3InferNetwork, self).__init__()
        self.network = network
        self.softmax = nn.Softmax(axis=1)
        self.format = input_format

    def construct(self, input_data):
        if self.format == "NHWC":
            input_data = ops.transpose(input_data, (0, 3, 1, 2))
        output = self.network(input_data)
        output = self.softmax(output)
        return output


class SoftmaxCrossEntropyLoss(nn.Cell):
    """
    softmax cross entropy loss
    """

    def __init__(self, num_cls=21, ignore_label=255):
        super(SoftmaxCrossEntropyLoss, self).__init__()
        self.cast = ops.Cast()
        self.one_hot = ops.OneHot(axis=-1)
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.ce = nn.SoftmaxCrossEntropyWithLogits()
        self.num_cls = num_cls
        self.ignore_label = ignore_label
        self.reduce_sum = ops.ReduceSum(False)

    def construct(self, logits, labels):
        labels_int = self.cast(labels, mstype.int32)
        labels_int = ops.reshape(labels_int, (-1,))
        logits_ = ops.transpose(logits, (0, 2, 3, 1))  # NCHW->NHWC
        logits_ = ops.reshape(logits_, (-1, self.num_cls))
        weights = ops.not_equal(labels_int, self.ignore_label)
        weights = self.cast(weights, mstype.float32)
        one_hot_labels = self.one_hot(labels_int, self.num_cls, self.on_value, self.off_value)
        loss = self.ce(logits_, one_hot_labels)
        loss = ops.mul(weights, loss)
        loss = ops.div(self.reduce_sum(loss), self.reduce_sum(weights))
        return loss
