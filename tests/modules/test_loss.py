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

ms.set_seed(1)
np.random.seed(1)

class SimpleCNN(nn.Cell):
    def __init__(self, num_classes=10, in_channels=1, include_top=True, aux_head=False, aux_head2=False):
        super(SimpleCNN, self).__init__()
        self.include_top = include_top
        self.aux_head = aux_head
        self.aux_head2 = aux_head2

        self.conv1 = nn.Conv2d(in_channels, 6, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
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

@pytest.mark.parametrize('mode', [0, 1])
@pytest.mark.parametrize('name', ['CE', 'BCE'])
@pytest.mark.parametrize('reduction', ['mean', 'sum'])
@pytest.mark.parametrize('label_smoothing', [0.0, 0.1])
@pytest.mark.parametrize('aux_factor', [0.0, 0.2])
@pytest.mark.parametrize('weight', [None, 1])
@pytest.mark.parametrize('double_aux', [False, True]) #TODO: decouple as one test case
def test_loss(mode, name, reduction, label_smoothing, aux_factor, weight, double_aux):
    weight=None
    print(f'mode={mode}; loss_name={name}; has_weight=False; reduction={reduction};\
        label_smoothing={label_smoothing}; aux_factor={aux_factor}')
    ms.set_context(mode=mode)

    bs = 8
    num_classes = c = 10
    # create data
    x = ms.Tensor(np.random.randn(bs, 1, 32, 32), ms.float32)
    #logits = ms.Tensor(np.random.rand(bs, c), ms.float32)
    y = np.random.randint(0, c, size=(bs))
    y_onehot = np.eye(c)[y]
    y = ms.Tensor(y, ms.int32) # C
    y_onehot = ms.Tensor(y_onehot, ms.float32) # N, C  
    if name=='BCE':
        label = y_onehot
    else:
        label = y

    if weight is not None:
        weight = np.random.randn(c)
        weight = weight / weight.sum() # normalize
        weight = ms.Tensor(weight, ms.float32) 

    # set network
    aux_head = aux_factor > 0.
    aux_head2 = aux_head and double_aux
    network = SimpleCNN(in_channels=1, num_classes=num_classes, aux_head=aux_head, aux_head2=aux_head2)
    
    # set loss
    net_loss = create_loss(name=name, weight=weight, reduction=reduction, label_smoothing=label_smoothing, aux_factor=aux_factor)
    
    # optimize
    net_with_loss = WithLossCell(network, net_loss)

    net_opt = create_optimizer(network.trainable_params(), 'adam', lr=0.001, weight_decay=1e-7)
    train_network = TrainOneStepCell(net_with_loss, net_opt)

    train_network.set_train()

    begin_loss = train_network(x, label)
    for _ in range(10):
        cur_loss = train_network(x, label)

    print("begin loss: {}, end loss:  {}".format(begin_loss, cur_loss))

    assert cur_loss < begin_loss, 'Loss does NOT decrease'


if __name__== '__main__':
    test_loss(0, 'BCE', 'mean', 0.1, 0.1, None, True)
