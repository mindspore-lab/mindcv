import sys

sys.path.append(".")

import numpy as np
import pytest

import mindspore as ms
from mindspore import Tensor, nn
from mindspore.common.initializer import Normal
from mindspore.nn import TrainOneStepCell, WithLossCell

from mindcv.loss import create_loss
from mindcv.optim import create_optimizer

ms.set_seed(1)
np.random.seed(1)


class SimpleCNN(nn.Cell):
    def __init__(self, num_classes=10, in_channels=1, include_top=True, aux_head=False, aux_head2=False):
        super(SimpleCNN, self).__init__()
        self.include_top = include_top
        self.aux_head = aux_head
        self.aux_head2 = aux_head2

        self.conv1 = nn.Conv2d(in_channels, 6, 5, pad_mode="valid")
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode="valid")
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)

        if self.include_top:
            self.flatten = nn.Flatten()
            self.fc = nn.Dense(16 * 5 * 5, num_classes, weight_init=Normal(0.02))

        if self.aux_head:
            self.fc_aux = nn.Dense(16 * 5 * 5, num_classes, weight_init=Normal(0.02))
        if self.aux_head:
            self.fc_aux2 = nn.Dense(16 * 5 * 5, num_classes, weight_init=Normal(0.02))

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
            if self.aux_head:
                x_aux = self.fc_aux(x_flatten)
                ret = (x, x_aux)
                if self.aux_head2:
                    x_aux2 = self.fc_aux2(x_flatten)
                    ret = (x, x_aux, x_aux2)
        return ret


@pytest.mark.parametrize("mode", [0, 1])
@pytest.mark.parametrize("name", ["CE", "BCE"])
@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("label_smoothing", [0.0, 0.1])
@pytest.mark.parametrize("aux_factor", [0.0, 0.2])
@pytest.mark.parametrize("weight", [None, 1])
@pytest.mark.parametrize("double_aux", [False, True])  # TODO: decouple as one test case
def test_loss(mode, name, reduction, label_smoothing, aux_factor, weight, double_aux):
    weight = None
    print(
        f"mode={mode}; loss_name={name}; has_weight=False; reduction={reduction};\
        label_smoothing={label_smoothing}; aux_factor={aux_factor}"
    )
    ms.set_context(mode=mode)

    bs = 8
    num_classes = c = 10
    # create data
    x = ms.Tensor(np.random.randn(bs, 1, 32, 32), ms.float32)
    # logits = ms.Tensor(np.random.rand(bs, c), ms.float32)
    y = np.random.randint(0, c, size=(bs))
    y_onehot = np.eye(c)[y]
    y = ms.Tensor(y, ms.int32)  # C
    y_onehot = ms.Tensor(y_onehot, ms.float32)  # N, C
    if name == "BCE":
        label = y_onehot
    else:
        label = y

    if weight is not None:
        weight = np.random.randn(c)
        weight = weight / weight.sum()  # normalize
        weight = ms.Tensor(weight, ms.float32)

    # set network
    aux_head = aux_factor > 0.0
    aux_head2 = aux_head and double_aux
    network = SimpleCNN(in_channels=1, num_classes=num_classes, aux_head=aux_head, aux_head2=aux_head2)

    # set loss
    net_loss = create_loss(
        name=name, weight=weight, reduction=reduction, label_smoothing=label_smoothing, aux_factor=aux_factor
    )

    # optimize
    net_with_loss = WithLossCell(network, net_loss)

    net_opt = create_optimizer(network.trainable_params(), "adam", lr=0.001, weight_decay=1e-7)
    train_network = TrainOneStepCell(net_with_loss, net_opt)

    train_network.set_train()

    begin_loss = train_network(x, label)
    for _ in range(10):
        cur_loss = train_network(x, label)

    print("begin loss: {}, end loss:  {}".format(begin_loss, cur_loss))

    assert cur_loss < begin_loss, "Loss does NOT decrease"


@pytest.mark.parametrize("mode", [0, 1])
@pytest.mark.parametrize("name", ["asl_single_label"])
@pytest.mark.parametrize("label_smoothing", [0.0, 0.1])
def test_asl(mode, name, label_smoothing, reduction="mean", aux_factor=0.0, weight=None, double_aux=False):
    weight = None
    print(
        f"mode={mode}; loss_name={name}; has_weight=False; reduction={reduction};\
        label_smoothing={label_smoothing}; aux_factor={aux_factor}"
    )
    ms.set_context(mode=mode)

    bs = 8
    num_classes = c = 10
    # create data
    x = ms.Tensor(np.random.randn(bs, 1, 32, 32), ms.float32)
    # logits = ms.Tensor(np.random.rand(bs, c), ms.float32)
    y = np.random.randint(0, c, size=(bs))
    y_onehot = np.eye(c)[y]
    y = ms.Tensor(y, ms.int32)  # C
    y_onehot = ms.Tensor(y_onehot, ms.float32)  # N, C
    if name == "BCE":
        label = y_onehot
    else:
        label = y

    if weight is not None:
        weight = np.random.randn(c)
        weight = weight / weight.sum()  # normalize
        weight = ms.Tensor(weight, ms.float32)

    # set network
    aux_head = aux_factor > 0.0
    aux_head2 = aux_head and double_aux
    network = SimpleCNN(in_channels=1, num_classes=num_classes, aux_head=aux_head, aux_head2=aux_head2)

    # set loss
    net_loss = create_loss(
        name=name, weight=weight, reduction=reduction, label_smoothing=label_smoothing, aux_factor=aux_factor
    )

    # optimize
    net_with_loss = WithLossCell(network, net_loss)

    net_opt = create_optimizer(network.trainable_params(), "adam", lr=0.001, weight_decay=1e-7)
    train_network = TrainOneStepCell(net_with_loss, net_opt)

    train_network.set_train()

    begin_loss = train_network(x, label)
    for _ in range(10):
        cur_loss = train_network(x, label)

    print("begin loss: {}, end loss:  {}".format(begin_loss, cur_loss))

    assert cur_loss < begin_loss, "Loss does NOT decrease"


@pytest.mark.parametrize("mode", [0, 1])
@pytest.mark.parametrize("name", ["asl_single_label"])
def test_asl_single_label_random(mode, name, reduction="mean", label_smoothing=0.1, aux_factor=0.0, weight=None):
    weight = None
    print(
        f"mode={mode}; loss_name={name}; has_weight=False; reduction={reduction};\
            label_smoothing={label_smoothing}; aux_factor={aux_factor}"
    )
    ms.set_context(mode=mode)

    net_loss = create_loss(
        name=name, weight=weight, reduction=reduction, label_smoothing=label_smoothing, aux_factor=aux_factor
    )

    # logits and labels
    logits = Tensor(
        [
            [0.38317937, 0.82873726, 0.8164871, 0.6443424],
            [0.77863216, 0.17288171, 0.69345415, 0.26514006],
            [0.14249292, 0.38524792, 0.97271717, 0.90531427],
        ],
        ms.float32,
    )
    labels = Tensor([1, 1, 1], ms.int32)

    output_expected = Tensor(1.1247127, ms.float32)

    output = net_loss(logits, labels)

    assert np.allclose(output_expected.asnumpy(), output.asnumpy())


@pytest.mark.parametrize("mode", [0, 1])
@pytest.mark.parametrize("name", ["asl_single_label"])
def test_asl_single_label_zero(mode, name, reduction="mean", label_smoothing=0.1, aux_factor=0.0, weight=None):
    weight = None
    print(
        f"mode={mode}; loss_name={name}; has_weight=False; reduction={reduction};\
            label_smoothing={label_smoothing}; aux_factor={aux_factor}"
    )
    ms.set_context(mode=mode)

    net_loss = create_loss(
        name=name, weight=weight, reduction=reduction, label_smoothing=label_smoothing, aux_factor=aux_factor
    )

    # logits and labels
    logits = Tensor(np.zeros((3, 5)), ms.float32)
    labels = Tensor(np.zeros((3,)), ms.int32)

    output_expected = Tensor(1.1847522, ms.float32)

    output = net_loss(logits, labels)

    assert np.allclose(output_expected.asnumpy(), output.asnumpy())


@pytest.mark.parametrize("mode", [0, 1])
@pytest.mark.parametrize("name", ["asl_multi_label"])
def test_asl_multi_label_random(mode, name, reduction="mean", label_smoothing=0.1, aux_factor=0.0, weight=None):
    weight = None
    print(
        f"mode={mode}; loss_name={name}; has_weight=False; reduction={reduction};\
            label_smoothing={label_smoothing}; aux_factor={aux_factor}"
    )
    ms.set_context(mode=mode)

    net_loss = create_loss(
        name=name, weight=weight, reduction=reduction, label_smoothing=label_smoothing, aux_factor=aux_factor
    )

    # logits and labels
    logits = Tensor(
        [
            [0.38317937, 0.82873726, 0.8164871, 0.6443424],
            [0.77863216, 0.17288171, 0.69345415, 0.26514006],
            [0.14249292, 0.38524792, 0.97271717, 0.90531427],
        ],
        ms.float32,
    )
    labels = Tensor([[1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 1, 0]], ms.int32)

    output_expected = Tensor(1.8657642, ms.float32)

    output = net_loss(logits, labels)

    assert np.allclose(output_expected.asnumpy(), output.asnumpy())


@pytest.mark.parametrize("mode", [0, 1])
@pytest.mark.parametrize("name", ["asl_multi_label"])
def test_asl_multi_label_zero(mode, name, reduction="mean", label_smoothing=0.1, aux_factor=0.0, weight=None):
    weight = None
    print(
        f"mode={mode}; loss_name={name}; has_weight=False; reduction={reduction};\
            label_smoothing={label_smoothing}; aux_factor={aux_factor}"
    )
    ms.set_context(mode=mode)

    net_loss = create_loss(
        name=name, weight=weight, reduction=reduction, label_smoothing=label_smoothing, aux_factor=aux_factor
    )

    # logits and labels
    logits = Tensor(np.zeros((3, 5)), ms.float32)
    labels = Tensor(np.zeros((3, 5)), ms.int32)

    output_expected = Tensor(0.3677258, ms.float32)

    output = net_loss(logits, labels)

    assert np.allclose(output_expected.asnumpy(), output.asnumpy())


if __name__ == "__main__":
    test_loss(0, "BCE", "mean", 0.1, 0.1, None, True)
