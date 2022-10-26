"""Test utils"""
import os
import sys
sys.path.append('.')
import pytest
import numpy as np

import mindspore as ms
from mindspore import Tensor, nn, ops
from mindspore.common.initializer import Normal
from mindspore.nn import TrainOneStepCell, WithLossCell

from mindcv.loss import create_loss
from mindcv.optim import create_optimizer
from mindcv.utils import CheckpointManager

ms.set_seed(1)
np.random.seed(1)

class SimpleCNN(nn.Cell):
    def __init__(self, num_classes=10, in_channels=1, include_top=True):
        super(SimpleCNN, self).__init__()
        self.include_top = include_top
        self.conv1 = nn.Conv2d(in_channels, 6, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
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

def validate(model, data, label):
    model.set_train(False)
    pred = model(data)
    total = len(data)
    acc = (pred.argmax(1) == label).sum()
    acc /= total
    return acc

@pytest.mark.parametrize('mode', [0, 1])
@pytest.mark.parametrize('ckpt_save_policy', ['top_k', 'latest_k'])
def test_checkpoint_manager(mode, ckpt_save_policy):
    ms.set_context(mode=mode)

    bs = 8
    num_classes = c = 10
    # create data
    x = ms.Tensor(np.random.randn(bs, 1, 32, 32), ms.float32)
    test_data = ms.Tensor(np.random.randn(bs, 1, 32, 32), ms.float32)
    test_label = ms.Tensor(np.random.randint(0, c, size=(bs)), ms.int32)
    y = np.random.randint(0, c, size=(bs))
    y = ms.Tensor(y, ms.int32)
    label = y

    network = SimpleCNN(in_channels=1, num_classes=num_classes)
    net_loss = create_loss()

    net_with_loss = WithLossCell(network, net_loss)
    net_opt = create_optimizer(network.trainable_params(), 'adam', lr=0.001, weight_decay=1e-7)
    train_network = TrainOneStepCell(net_with_loss, net_opt)
    train_network.set_train()
    manager = CheckpointManager(ckpt_save_policy=ckpt_save_policy)
    for t in range(3):
        train_network(x, label)
        acc = validate(network, test_data, test_label)
        save_path = os.path.join('./' + f'network_{t + 1}.ckpt')
        ckpoint_filelist = manager.save_ckpoint(network, num_ckpt=2, metric=acc, save_path=save_path)

    assert len(ckpoint_filelist) == 2