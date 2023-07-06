import sys

import numpy as np
import pytest

import mindspore as ms
import mindspore.nn as nn

sys.path.append(".")

from mindcv.models import create_model
from mindcv.models.features import FeatureExtractWrapper


class Conv2dReLU(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size, downsample=False):
        super().__init__()
        self.downsample = downsample
        if self.downsample:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, pad_mode="valid")
        self.relu = nn.ReLU()

    def construct(self, x):
        if self.downsample:
            x = self.pool(x)
        x = self.conv(x)
        x = self.relu(x)
        return x


class SimpleCNN(nn.Cell):
    def __init__(self, in_channels=1):
        super(SimpleCNN, self).__init__()
        self.block1 = Conv2dReLU(in_channels, 6, 5, False)
        self.block2 = Conv2dReLU(6, 8, 5, True)
        self.block3 = Conv2dReLU(8, 8, 5, True)

        self.feature_info = [
            {"chs": 6, "reduction": 1, "name": "block1"},
            {"chs": 8, "reduction": 2, "name": "block2"},
            {"chs": 8, "reduction": 4, "name": "block3"},
        ]

    def forward_features(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x

    def forward_multi_features(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        return [x1, x2, x3]

    def construct(self, x):
        return self.forward_features(x)


class SimpleCNNWithInnerSequential(nn.Cell):
    def __init__(self, in_channels=1):
        super(SimpleCNNWithInnerSequential, self).__init__()
        block1 = Conv2dReLU(in_channels, 6, 5, False)
        block2 = Conv2dReLU(6, 8, 5, True)
        block3 = Conv2dReLU(8, 8, 5, True)
        self.feature = nn.SequentialCell([block1, block2, block3])

        self.feature_info = [
            {"chs": 6, "reduction": 1, "name": "feature.0"},
            {"chs": 8, "reduction": 2, "name": "feature.1"},
            {"chs": 8, "reduction": 4, "name": "feature.2"},
        ]
        self.flatten_sequential = True

    def forward_features(self, x):
        x = self.feature(x)
        return x

    def forward_multi_features(self, x):
        x1 = self.feature[0](x)
        x2 = self.feature[1](x1)
        x3 = self.feature[2](x2)
        return [x1, x2, x3]

    def construct(self, x):
        return self.forward_features(x)


def output_feature(args):
    network = create_model(
        model_name=args.model,
        num_classes=args.num_classes,
        drop_rate=args.drop_rate,
        drop_path_rate=args.drop_path_rate,
        pretrained=args.pretrained,
        checkpoint_path=args.ckpt_path,
        ema=args.ema,
        features_only=True,
    )

    assert isinstance(network, nn.Cell), "Loading checkpoint error"

    x = ms.Tensor(np.random.randn(8, 3, 32, 32), dtype=ms.float32)
    out = network(x)

    return len(out)


@pytest.mark.parametrize("mode", [0, 1])
def test_feature_extraction_result_using_feature_wrapper(mode):
    ms.set_context(mode=mode)
    np.random.seed(0)

    net = SimpleCNN()
    x = ms.Tensor(np.random.randn(8, 1, 32, 32), dtype=ms.float32)
    y0 = net.forward_multi_features(x)

    wrapped_net = FeatureExtractWrapper(net, [0, 1, 2])
    y1 = wrapped_net(x)

    assert len(y0) == len(y1)
    assert len(y1) == 3
    for z0, z1 in zip(y0, y1):
        np.testing.assert_equal(z1.asnumpy(), z0.asnumpy())


@pytest.mark.parametrize("mode", [0, 1])
def test_feature_extraction_result_using_feature_wrapper_with_flatten_sequential(mode):
    ms.set_context(mode=mode)
    np.random.seed(0)

    net = SimpleCNNWithInnerSequential()
    x = ms.Tensor(np.random.randn(8, 1, 32, 32), dtype=ms.float32)
    y0 = net.forward_multi_features(x)

    wrapped_net = FeatureExtractWrapper(net, [0, 1, 2])
    y1 = wrapped_net(x)

    assert len(y0) == len(y1)
    assert len(y1) == 3
    for z0, z1 in zip(y0, y1):
        np.testing.assert_equal(z1.asnumpy(), z0.asnumpy())


@pytest.mark.parametrize("mode", [0, 1])
def test_feature_extraction_indices_using_feature_wrapper(mode):
    ms.set_context(mode=mode)
    np.random.seed(0)

    net = SimpleCNN()
    x = ms.Tensor(np.random.randn(8, 1, 32, 32), dtype=ms.float32)
    wrapped_net = FeatureExtractWrapper(net, [0, 1, 2])
    y0 = wrapped_net(x)
    assert len(y0) == 3

    wrapped_net2 = FeatureExtractWrapper(net, [1])
    y1 = wrapped_net2(x)
    assert len(y1) == 1

    np.testing.assert_equal(y0[1].asnumpy(), y1[0].asnumpy())


@pytest.mark.parametrize(
    "model_name, length_target",
    [
        (
            "resnet18",
            5,
        ),
        (
            "mobilenet_v3_small_100",
            5,
        ),
        (
            "convnext_tiny",
            4,
        ),
        (
            "resnest50",
            5,
        ),
        (
            "efficientnet_b0",
            5,
        ),
        (
            "repvgg_a0",
            5,
        ),
        (
            "hrnet_w32",
            5,
        ),
        (
            "rexnet_10",
            5,
        ),
    ],
)
def test_feature_extraction_with_checkpoint(model_name, length_target):
    model = create_model(
        model_name=model_name,
        pretrained=True,
        features_only=True,
    )

    assert isinstance(model, nn.Cell), "Loading checkpoint error"

    x = ms.Tensor(np.random.randn(8, 3, 32, 32), dtype=ms.float32)
    out = model(x)

    assert len(out) == length_target, "Wrong feature extraction"
