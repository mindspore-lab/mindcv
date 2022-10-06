import sys

sys.path.append('.')

import pytest

from mindspore.communication import init, get_rank, get_group_size
from mindspore import Tensor
import mindspore as ms
import mindspore.nn as nn
from mindspore.nn import TrainOneStepCell, WithLossCell
from mindspore.common.initializer import Normal

from mindcv.optim import create_optimizer
import numpy as np


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
        if self.include_top:
            x = self.flatten(x)
            x = self.fc(x)
        return x


@pytest.mark.parametrize('opt', ['sgd', 'momentum'])
@pytest.mark.parametrize('nesterov', [True, False])
@pytest.mark.parametrize('filter_bias_and_bn', [True, False])
def test_sgd_optimizer(opt, nesterov, filter_bias_and_bn):
    init("nccl")
    device_num = get_group_size()
    rank_id = get_rank()
    ms.set_auto_parallel_context(device_num=device_num,
                                 parallel_mode='data_parallel',
                                 gradients_mean=True)
    network = SimpleCNN(in_channels=1, num_classes=10)
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    net_opt = create_optimizer(network.trainable_params(), opt, lr=0.01,
                               weight_decay=1e-5, momentum=0.9, nesterov=nesterov,
                               filter_bias_and_bn=filter_bias_and_bn)

    bs = 8
    input_data = Tensor(np.ones([bs, 1, 32, 32]).astype(np.float32) * 0.01)
    label = Tensor(np.ones([bs]).astype(np.int32))

    net_with_loss = WithLossCell(network, net_loss)
    train_network = TrainOneStepCell(net_with_loss, net_opt)

    train_network.set_train()

    begin_loss = train_network(input_data, label)
    for i in range(10):
        cur_loss = train_network(input_data, label)
    print(f"{opt}, begin loss: {begin_loss}, end loss:  {cur_loss}")

    # check output correctness
    assert cur_loss < begin_loss, 'Loss does NOT decrease'


@pytest.mark.parametrize('bs', [1, 2, 4, 8, 16])
@pytest.mark.parametrize('opt', ['adam', 'adamW', 'rmsprop', 'adagrad'])
def test_bs_adam_optimizer(opt, bs):
    init("nccl")
    device_num = get_group_size()
    rank_id = get_rank()
    ms.set_auto_parallel_context(device_num=device_num,
                                 parallel_mode='data_parallel',
                                 gradients_mean=True)
    network = SimpleCNN(num_classes=10)
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    net_opt = create_optimizer(network.trainable_params(), opt, lr=0.01, weight_decay=1e-5)

    bs = bs
    input_data = Tensor(np.ones([bs, 1, 32, 32]).astype(np.float32) * 0.01)
    label = Tensor(np.ones([bs]).astype(np.int32))

    net_with_loss = WithLossCell(network, net_loss)
    train_network = TrainOneStepCell(net_with_loss, net_opt)

    train_network.set_train()

    begin_loss = train_network(input_data, label)
    for i in range(10):
        cur_loss = train_network(input_data, label)

    print(f"{opt}, begin loss: {begin_loss}, end loss:  {cur_loss}")

    # check output correctness
    assert cur_loss < begin_loss, 'Loss does NOT decrease'


@pytest.mark.parametrize('loss_scale', [0.1, 0.2, 0.3, 0.5, 0.9, 1.0])
@pytest.mark.parametrize('weight_decay', [0.00001, 0.0001, 0.001, 0.005, 0.01, 0.05])
@pytest.mark.parametrize('lr', [0.0001, 0.001, 0.005, 0.05, 0.1, 0.2])
def test_lr_weight_decay_loss_scale_optimizer(lr, weight_decay, loss_scale):
    init("nccl")
    device_num = get_group_size()
    rank_id = get_rank()
    ms.set_auto_parallel_context(device_num=device_num,
                                 parallel_mode='data_parallel',
                                 gradients_mean=True)
    network = SimpleCNN(num_classes=10)
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    net_opt = create_optimizer(network.trainable_params(), 'adamW', lr=lr, weight_decay=weight_decay,
                               loss_scale=loss_scale)

    bs = 8
    input_data = Tensor(np.ones([bs, 1, 32, 32]).astype(np.float32) * 0.01)
    label = Tensor(np.ones([bs]).astype(np.int32))

    net_with_loss = WithLossCell(network, net_loss)
    train_network = TrainOneStepCell(net_with_loss, net_opt)

    train_network.set_train()

    begin_loss = train_network(input_data, label)
    for i in range(10):
        cur_loss = train_network(input_data, label)

    print(f"{lr}, {weight_decay}, {loss_scale}, begin loss: {begin_loss}, end loss:  {cur_loss}")

    # check output correctness
    assert cur_loss < begin_loss, 'Loss does NOT decrease'


@pytest.mark.parametrize('momentum', [0.1, 0.2, 0.5, 0.9, 0.99])
def test_momentum_optimizer(momentum):
    init("nccl")
    device_num = get_group_size()
    rank_id = get_rank()
    ms.set_auto_parallel_context(device_num=device_num,
                                 parallel_mode='data_parallel',
                                 gradients_mean=True)
    network = SimpleCNN(in_channels=1, num_classes=10)
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    net_opt = create_optimizer(network.trainable_params(), 'momentum', lr=0.01, weight_decay=1e-5, momentum=momentum,
                               nesterov=False)

    bs = 8
    input_data = Tensor(np.ones([bs, 1, 32, 32]).astype(np.float32) * 0.01)
    label = Tensor(np.ones([bs]).astype(np.int32))

    net_with_loss = WithLossCell(network, net_loss)
    train_network = TrainOneStepCell(net_with_loss, net_opt)

    train_network.set_train()

    begin_loss = train_network(input_data, label)
    for i in range(10):
        cur_loss = train_network(input_data, label)
    print(f"{momentum}, begin loss: {begin_loss}, end loss:  {cur_loss}")

    # check output correctness
    assert cur_loss < begin_loss, 'Loss does NOT decrease'


def test_param_lr_001_filter_bias_and_bn_optimizer():
    init("nccl")
    device_num = get_group_size()
    rank_id = get_rank()
    ms.set_auto_parallel_context(device_num=device_num,
                                 parallel_mode='data_parallel',
                                 gradients_mean=True)
    network = SimpleCNN(in_channels=1, num_classes=10)
    conv_params = list(filter(lambda x: 'conv' in x.name, network.trainable_params()))
    no_conv_params = list(filter(lambda x: 'conv' not in x.name, network.trainable_params()))
    group_params = [{'params': conv_params, 'weight_decay': 0.01, 'grad_centralization': True},
                    {'params': no_conv_params, 'lr': 0.01},
                    {'order_params': network.trainable_params()}]
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    net_opt = create_optimizer(group_params, 'adamW', lr=0.01, weight_decay=1e-5, momentum=0.9,
                               nesterov=False, filter_bias_and_bn=False)

    bs = 8
    input_data = Tensor(np.ones([bs, 1, 32, 32]).astype(np.float32) * 0.01)
    label = Tensor(np.ones([bs]).astype(np.int32))

    net_with_loss = WithLossCell(network, net_loss)
    train_network = TrainOneStepCell(net_with_loss, net_opt)

    train_network.set_train()

    begin_loss = train_network(input_data, label)
    for i in range(10):
        cur_loss = train_network(input_data, label)
    print(f" begin loss: {begin_loss}, end loss:  {cur_loss}")

    # check output correctness
    assert cur_loss < begin_loss, 'Loss does NOT decrease'


def test_param_lr_0001_filter_bias_and_bn_optimizer():
    init("nccl")
    device_num = get_group_size()
    rank_id = get_rank()
    ms.set_auto_parallel_context(device_num=device_num,
                                 parallel_mode='data_parallel',
                                 gradients_mean=True)
    network = SimpleCNN(in_channels=1, num_classes=10)
    conv_params = list(filter(lambda x: 'conv' in x.name, network.trainable_params()))
    no_conv_params = list(filter(lambda x: 'conv' not in x.name, network.trainable_params()))
    group_params = [{'params': conv_params, 'weight_decay': 0.01, 'grad_centralization': True},
                    {'params': no_conv_params, 'lr': 0.001},
                    {'order_params': network.trainable_params()}]
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    net_opt = create_optimizer(group_params, 'adamW', lr=0.01, weight_decay=1e-5, momentum=0.9,
                               nesterov=False, filter_bias_and_bn=False)

    bs = 8
    input_data = Tensor(np.ones([bs, 1, 32, 32]).astype(np.float32) * 0.01)
    label = Tensor(np.ones([bs]).astype(np.int32))

    net_with_loss = WithLossCell(network, net_loss)
    train_network = TrainOneStepCell(net_with_loss, net_opt)

    train_network.set_train()

    begin_loss = train_network(input_data, label)
    for i in range(10):
        cur_loss = train_network(input_data, label)
    print(f" begin loss: {begin_loss}, end loss:  {cur_loss}")

    # check output correctness
    assert cur_loss < begin_loss, 'Loss does NOT decrease'


@pytest.mark.parametrize('momentum', [-0.1, -1.0, -2])
def test_wrong_momentum_optimizer(momentum):
    init("nccl")
    device_num = get_group_size()
    rank_id = get_rank()
    ms.set_auto_parallel_context(device_num=device_num,
                                 parallel_mode='data_parallel',
                                 gradients_mean=True)
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        network = SimpleCNN(in_channels=1, num_classes=10)
        net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        net_opt = create_optimizer(network.trainable_params(), 'momentum', lr=0.01,
                                   weight_decay=0.0001, momentum=momentum, loss_scale=1.0,
                                   nesterov=False, filter_bias_and_bn=True)

        bs = 8
        input_data = Tensor(np.ones([bs, 1, 32, 32]).astype(np.float32) * 0.01)
        label = Tensor(np.ones([bs]).astype(np.int32))

        net_with_loss = WithLossCell(network, net_loss)
        train_network = TrainOneStepCell(net_with_loss, net_opt)

        train_network.set_train()

        begin_loss = train_network(input_data, label)
        for i in range(10):
            cur_loss = train_network(input_data, label)
        print(f"{momentum}, begin loss: {begin_loss}, end loss:  {cur_loss}")

        # check output correctness
        assert cur_loss < begin_loss, 'Loss does NOT decrease'


@pytest.mark.parametrize('loss_scale', [-0.1, -1.0])
def test_wrong_loss_scale_optimizer(loss_scale):
    init("nccl")
    device_num = get_group_size()
    rank_id = get_rank()
    ms.set_auto_parallel_context(device_num=device_num,
                                 parallel_mode='data_parallel',
                                 gradients_mean=True)
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        network = SimpleCNN(in_channels=1, num_classes=10)
        net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        net_opt = create_optimizer(network.trainable_params(), 'momentum', lr=0.01,
                                   weight_decay=0.0001, momentum=0.9, loss_scale=loss_scale,
                                   nesterov=False, filter_bias_and_bn=True)

        bs = 8
        input_data = Tensor(np.ones([bs, 1, 32, 32]).astype(np.float32) * 0.01)
        label = Tensor(np.ones([bs]).astype(np.int32))

        net_with_loss = WithLossCell(network, net_loss)
        train_network = TrainOneStepCell(net_with_loss, net_opt)

        train_network.set_train()

        begin_loss = train_network(input_data, label)
        for i in range(10):
            cur_loss = train_network(input_data, label)
        print(f"{loss_scale}, begin loss: {begin_loss}, end loss:  {cur_loss}")

        # check output correctness
        if cur_loss < begin_loss:
            raise ValueError


@pytest.mark.parametrize('weight_decay', [-0.1, -1.0])
def test_wrong_weight_decay_optimizer(weight_decay):
    init("nccl")
    device_num = get_group_size()
    rank_id = get_rank()
    ms.set_auto_parallel_context(device_num=device_num,
                                 parallel_mode='data_parallel',
                                 gradients_mean=True)
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        network = SimpleCNN(in_channels=1, num_classes=10)
        net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        net_opt = create_optimizer(network.trainable_params(), 'adamW', lr=0.01,
                                   weight_decay=weight_decay, momentum=0.9, loss_scale=1.0,
                                   nesterov=False, filter_bias_and_bn=True)

        bs = 8
        input_data = Tensor(np.ones([bs, 1, 32, 32]).astype(np.float32) * 0.01)
        label = Tensor(np.ones([bs]).astype(np.int32))

        net_with_loss = WithLossCell(network, net_loss)
        train_network = TrainOneStepCell(net_with_loss, net_opt)

        train_network.set_train()

        begin_loss = train_network(input_data, label)
        for i in range(10):
            cur_loss = train_network(input_data, label)
        print(f"{weight_decay}, begin loss: {begin_loss}, end loss:  {cur_loss}")

        # check output correctness
        assert cur_loss < begin_loss, 'Loss does NOT decrease'


@pytest.mark.parametrize('lr', [-1.0, -0.1])
def test_wrong_lr_optimizer(lr):
    init("nccl")
    device_num = get_group_size()
    rank_id = get_rank()
    ms.set_auto_parallel_context(device_num=device_num,
                                 parallel_mode='data_parallel',
                                 gradients_mean=True)
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        network = SimpleCNN(in_channels=1, num_classes=10)
        net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        net_opt = create_optimizer(network.trainable_params(), 'adamW', lr=lr,
                                   weight_decay=1e-5, momentum=0.9, loss_scale=1.0,
                                   nesterov=False, filter_bias_and_bn=True)

        bs = 8
        input_data = Tensor(np.ones([bs, 1, 32, 32]).astype(np.float32) * 0.01)
        label = Tensor(np.ones([bs]).astype(np.int32))

        net_with_loss = WithLossCell(network, net_loss)
        train_network = TrainOneStepCell(net_with_loss, net_opt)

        train_network.set_train()

        begin_loss = train_network(input_data, label)
        for i in range(10):
            cur_loss = train_network(input_data, label)
        print(f"{lr}, begin loss: {begin_loss}, end loss:  {cur_loss}")

        # check output correctness
        assert cur_loss < begin_loss, 'Loss does NOT decrease'


def test_param_lr_01_filter_bias_and_bn_optimizer():
    init("nccl")
    device_num = get_group_size()
    rank_id = get_rank()
    ms.set_auto_parallel_context(device_num=device_num,
                                 parallel_mode='data_parallel',
                                 gradients_mean=True)
    network = SimpleCNN(in_channels=1, num_classes=10)
    conv_params = list(filter(lambda x: 'conv' in x.name, network.trainable_params()))
    no_conv_params = list(filter(lambda x: 'conv' not in x.name, network.trainable_params()))
    group_params = [{'params': conv_params, 'weight_decay': 0.01, 'grad_centralization': True},
                    {'params': no_conv_params, 'lr': 0.1},
                    {'order_params': network.trainable_params()}]
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    net_opt = create_optimizer(group_params, 'momentum', lr=0.01, weight_decay=1e-5, momentum=0.9,
                               nesterov=False, filter_bias_and_bn=False)

    bs = 8
    input_data = Tensor(np.ones([bs, 1, 32, 32]).astype(np.float32) * 0.01)
    label = Tensor(np.ones([bs]).astype(np.int32))

    net_with_loss = WithLossCell(network, net_loss)
    train_network = TrainOneStepCell(net_with_loss, net_opt)

    train_network.set_train()

    begin_loss = train_network(input_data, label)
    for i in range(10):
        cur_loss = train_network(input_data, label)
    print(f" begin loss: {begin_loss}, end loss:  {cur_loss}")

    # check output correctness
    assert cur_loss < begin_loss, 'Loss does NOT decrease'


@pytest.mark.parametrize('opt', ['test', 'bdam', 'mindspore'])
def test_wrong_opt_optimizer(opt):
    init("nccl")
    device_num = get_group_size()
    rank_id = get_rank()
    ms.set_auto_parallel_context(device_num=device_num,
                                 parallel_mode='data_parallel',
                                 gradients_mean=True)
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        network = SimpleCNN(in_channels=1, num_classes=10)
        net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        net_opt = create_optimizer(network.trainable_params(), opt, lr=0.01,
                                   weight_decay=1e-5, momentum=0.9, loss_scale=1.0,
                                   nesterov=False, filter_bias_and_bn=True)

        bs = 8
        input_data = Tensor(np.ones([bs, 1, 32, 32]).astype(np.float32) * 0.01)
        label = Tensor(np.ones([bs]).astype(np.int32))

        net_with_loss = WithLossCell(network, net_loss)
        train_network = TrainOneStepCell(net_with_loss, net_opt)

        train_network.set_train()

        begin_loss = train_network(input_data, label)
        for i in range(10):
            cur_loss = train_network(input_data, label)
        print(f"{opt}, begin loss: {begin_loss}, end loss:  {cur_loss}")

        # check output correctness
        assert cur_loss < begin_loss, 'Loss does NOT decrease'


def test_wrong_params_more_optimizer():
    init("nccl")
    device_num = get_group_size()
    rank_id = get_rank()
    ms.set_auto_parallel_context(device_num=device_num,
                                 parallel_mode='data_parallel',
                                 gradients_mean=True)
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        network = SimpleCNN(in_channels=1, num_classes=10)
        conv_params = list(filter(lambda x: 'conv' in x.name, network.trainable_params()))
        conv_params.append('test')
        no_conv_params = list(filter(lambda x: 'conv' not in x.name, network.trainable_params()))
        group_params = [{'params': conv_params, 'weight_decay': 0.01, 'grad_centralization': True},
                        {'params': no_conv_params, 'lr': 0.0},
                        {'order_params': network.trainable_params()}]
        net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        net_opt = create_optimizer(group_params, 'momentum', lr=0.01,
                                   weight_decay=1e-5, momentum=0.9, loss_scale=1.0,
                                   nesterov=False, filter_bias_and_bn=False)

        bs = 8
        input_data = Tensor(np.ones([bs, 1, 32, 32]).astype(np.float32) * 0.01)
        label = Tensor(np.ones([bs]).astype(np.int32))

        net_with_loss = WithLossCell(network, net_loss)
        train_network = TrainOneStepCell(net_with_loss, net_opt)

        train_network.set_train()

        begin_loss = train_network(input_data, label)
        for i in range(10):
            cur_loss = train_network(input_data, label)
        print(f" begin loss: {begin_loss}, end loss:  {cur_loss}")

        # check output correctness
        assert cur_loss < begin_loss, 'Loss does NOT decrease'


def test_wrong_params_input_optimizer():
    init("nccl")
    device_num = get_group_size()
    rank_id = get_rank()
    ms.set_auto_parallel_context(device_num=device_num,
                                 parallel_mode='data_parallel',
                                 gradients_mean=True)
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        network = SimpleCNN(in_channels=1, num_classes=10)
        conv_params = [1, 2, 3, 4]
        no_conv_params = list(filter(lambda x: 'conv' not in x.name, network.trainable_params()))
        group_params = [{'params': conv_params, 'weight_decay': 0.01, 'grad_centralization': True},
                        {'params': no_conv_params, 'lr': 0.0},
                        {'order_params': network.trainable_params()}]
        net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        net_opt = create_optimizer(group_params, 'momentum', lr=0.01,
                                   weight_decay=1e-5, momentum=0.9, loss_scale=1.0,
                                   nesterov=False, filter_bias_and_bn=False)

        bs = 8
        input_data = Tensor(np.ones([bs, 1, 32, 32]).astype(np.float32) * 0.01)
        label = Tensor(np.ones([bs]).astype(np.int32))

        net_with_loss = WithLossCell(network, net_loss)
        train_network = TrainOneStepCell(net_with_loss, net_opt)

        train_network.set_train()

        begin_loss = train_network(input_data, label)
        for i in range(10):
            cur_loss = train_network(input_data, label)
        print(f" begin loss: {begin_loss}, end loss:  {cur_loss}")

        # check output correctness
        assert cur_loss < begin_loss, 'Loss does NOT decrease'


@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE, ])
def test_mode_mult_single_optimizer(mode):
    init("nccl")
    device_num = get_group_size()
    rank_id = get_rank()
    ms.set_auto_parallel_context(device_num=device_num,
                                 parallel_mode='data_parallel',
                                 gradients_mean=True)
    ms.set_context(mode=mode)
    network = SimpleCNN(in_channels=1, num_classes=10)
    conv_params = list(filter(lambda x: 'conv' in x.name, network.trainable_params()))
    no_conv_params = list(filter(lambda x: 'conv' not in x.name, network.trainable_params()))
    group_params = [{'params': conv_params, 'weight_decay': 0.01, 'grad_centralization': True},
                    {'params': no_conv_params, 'lr': 0.1},
                    {'order_params': network.trainable_params()}]
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    net_opt = create_optimizer(group_params, 'momentum', lr=0.01, weight_decay=1e-5, momentum=0.9,
                               nesterov=False, filter_bias_and_bn=False)

    bs = 8
    input_data = Tensor(np.ones([bs, 1, 32, 32]).astype(np.float32) * 0.01)
    label = Tensor(np.ones([bs]).astype(np.int32))

    net_with_loss = WithLossCell(network, net_loss)
    train_network = TrainOneStepCell(net_with_loss, net_opt)

    train_network.set_train()

    begin_loss = train_network(input_data, label)
    for i in range(10):
        cur_loss = train_network(input_data, label)
    print(f" begin loss: {begin_loss}, end loss:  {cur_loss}")

    # check output correctness
    assert cur_loss < begin_loss, 'Loss does NOT decrease'
