import sys
sys.path.append('.')
import pytest
import numpy as np

import mindspore as ms
from mindspore import Tensor, nn
from mindspore.common.initializer import Normal
from mindspore.nn import TrainOneStepCell, WithLossCell

from mindcv.loss import create_loss
from mindcv.optim import create_optimizer

class SimpleCNN(nn.Cell):
    def __init__(self, num_classes=10, in_channels=1, include_top=True, aux_head=False):
        super(SimpleCNN, self).__init__()
        self.include_top = include_top
        self.aux_head = aux_head

        self.conv1 = nn.Conv2d(in_channels, 6, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)

        if self.include_top:
            self.flatten = nn.Flatten()
            self.fc = nn.Dense(16 * 5 * 5, num_classes, weight_init=Normal(0.02))

        if self.aux_head:
            self.fc_aux = nn.Dense(16 * 5 * 5, num_classes, weight_init=Normal(0.02))

    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        if self.include_top:
            x_flatten = self.flatten(x)
            x = self.fc(x_flatten)
            if self.aux_head:
                x_aux = self.fc_aux(x_flatten)
                return x, x_aux
        return x

@pytest.mark.parametrize('mode', [0, 1])
@pytest.mark.parametrize('name', ['CE', 'BCE'])
@pytest.mark.parametrize('reduction', ['mean', 'sum', 'none'])
@pytest.mark.parametrize('label_smoothing', [0.0, 0.1])
@pytest.mark.parametrize('aux_factor', [0.0, 0.2])
def test_loss(mode, name, reduction, label_smoothing, aux_factor):
    weight=None
    print(f'mode={mode}; loss_name={name}; has_weight=False; reduction={reduction};\
        label_smoothing={label_smoothing}; aux_factor={aux_factor}')
    ms.set_context(mode=mode)
    num_classes = 10
    aux_head = False
    if aux_factor:
        aux_head = True
    network = SimpleCNN(in_channels=1, num_classes=num_classes, aux_head=aux_head)
    
    net_opt = create_optimizer(network.trainable_params(), 'momentum', lr=0.01, weight_decay=1e-5, momentum=0.9, nesterov=False)
    net_loss = create_loss(name=name, weight=weight, reduction=reduction, label_smoothing=label_smoothing, aux_factor=aux_factor)
    bs = 8
    input_data = Tensor(np.random.randn(bs, 1, 32, 32).astype(np.float32))
    if name=='CE':
        label = Tensor(np.random.randint(0, high=1, size=(bs)).astype(np.int32))
    else:
        label = Tensor(np.random.randint(0, high=1, size=(bs, num_classes)).astype(np.float32))
    net_with_loss = WithLossCell(network, net_loss)
    train_network = TrainOneStepCell(net_with_loss, net_opt)

    train_network.set_train()

    begin_loss = train_network(input_data, label)
    for _ in range(10):
        cur_loss = train_network(input_data, label)

    print("begin loss: {}, end loss:  {}".format(begin_loss, cur_loss))

    assert cur_loss < begin_loss, 'Loss does NOT decrease'

@pytest.mark.parametrize('mode', [0, 1])
@pytest.mark.parametrize('name', ['CE', 'BCE'])
@pytest.mark.parametrize('reduction', ['mean', 'sum', 'none'])
@pytest.mark.parametrize('label_smoothing', [0.0, 0.1])
@pytest.mark.parametrize('aux_factor', [0.0, 0.2])
def test_loss_with_weight(mode, name, reduction, label_smoothing, aux_factor):
    print(f'mode={mode}; loss_name={name}; has_weight=True; reduction={reduction};\
        label_smoothing={label_smoothing}; aux_factor={aux_factor}')
    ms.set_context(mode=mode)
    num_classes = 10
    weight = Tensor(np.random.randn(num_classes), dtype=ms.float32)
    aux_head = False
    if aux_factor:
        aux_head = True
    network = SimpleCNN(in_channels=1, num_classes=num_classes, aux_head=aux_head)
    
    net_opt = create_optimizer(network.trainable_params(), 'momentum', lr=0.01, weight_decay=1e-5, momentum=0.9, nesterov=False)
    net_loss = create_loss(name=name, weight=weight, reduction=reduction, label_smoothing=label_smoothing, aux_factor=aux_factor)
    bs = 8
    input_data = Tensor(np.random.randn(bs, 1, 32, 32).astype(np.float32))
    if name=='CE':
        label = Tensor(np.random.randint(0, high=1, size=(bs)).astype(np.int32))
    else:
        label = Tensor(np.random.randint(0, high=1, size=(bs, num_classes)).astype(np.float32))
    net_with_loss = WithLossCell(network, net_loss)
    train_network = TrainOneStepCell(net_with_loss, net_opt)

    train_network.set_train()

    begin_loss = train_network(input_data, label)
    for _ in range(10):
        cur_loss = train_network(input_data, label)

    print("begin loss: {}, end loss:  {}".format(begin_loss, cur_loss))

    assert cur_loss < begin_loss, 'Loss does NOT decrease'
