"""Test utils"""
import sys

sys.path.append(".")

import numpy as np
import pytest

import mindspore as ms
from mindspore import Tensor, nn
from mindspore.common.initializer import Normal
from mindspore.nn import WithLossCell

from mindcv.optim import create_optimizer
from mindcv.utils import TrainStep

ms.set_seed(1)
np.random.seed(1)


class SimpleCNN(nn.Cell):
    def __init__(self, num_classes=10, in_channels=1, include_top=True):
        super(SimpleCNN, self).__init__()
        self.include_top = include_top
        self.conv1 = nn.Conv2d(in_channels, 6, 5, pad_mode="valid")
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode="valid")
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)

        if self.include_top:
            self.flatten = nn.Flatten()
            self.fc = nn.Dense(16 * 5 * 5, num_classes, weight_init=Normal(0.02))

    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        ret = x
        if self.include_top:
            x_flatten = self.flatten(x)
            x = self.fc(x_flatten)
            ret = x
        return ret


@pytest.mark.parametrize("ema", [True, False])
@pytest.mark.parametrize("ema_decay", [0.9997, 0.5])
def test_ema(ema, ema_decay):
    network = SimpleCNN(in_channels=1, num_classes=10)
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")

    net_opt = create_optimizer(network.trainable_params(), "adam", lr=0.001, weight_decay=1e-7)

    bs = 8
    input_data = Tensor(np.ones([bs, 1, 32, 32]).astype(np.float32) * 0.01)
    label = Tensor(np.ones([bs]).astype(np.int32))

    net_with_loss = WithLossCell(network, net_loss)
    loss_scale_manager = Tensor(1, ms.float32)
    train_network = TrainStep(net_with_loss, net_opt, scale_sense=loss_scale_manager, ema=ema, ema_decay=ema_decay)
    train_network.set_train()

    begin_loss = train_network(input_data, label)
    for i in range(10):
        cur_loss = train_network(input_data, label)
    print(f"{net_opt}, begin loss: {begin_loss}, end loss:  {cur_loss}")

    # check output correctness
    assert cur_loss < begin_loss, "Loss does NOT decrease"


@pytest.mark.parametrize("gradient_accumulation_steps", [1, 4])
def test_gradient_accumulation(gradient_accumulation_steps):
    network = SimpleCNN(in_channels=1, num_classes=10)
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")

    net_opt = create_optimizer(network.trainable_params(), "adam", lr=0.001, weight_decay=1e-7)

    bs = 8
    input_data = Tensor(np.ones([bs, 1, 32, 32]).astype(np.float32) * 0.01)
    label = Tensor(np.ones([bs]).astype(np.int32))

    net_with_loss = WithLossCell(network, net_loss)
    loss_scale_manager = Tensor(1, ms.float32)
    train_network = TrainStep(
        net_with_loss, net_opt, scale_sense=loss_scale_manager, gradient_accumulation_steps=gradient_accumulation_steps
    )
    train_network.set_train()

    # For accumulation batches = N, the weight is not updated for N - 1 steps,
    # Thus the loss should be same when data input is same.
    begin_loss = train_network(input_data, label)
    for _ in range(gradient_accumulation_steps - 1):
        cur_loss = train_network(input_data, label)
        assert cur_loss == begin_loss
    cur_loss = train_network(input_data, label)
    assert cur_loss < begin_loss
