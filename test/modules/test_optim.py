import sys
sys.path.append('.')

import pytest

from mindcv.optim import create_optimizer
import numpy as np
from mindspore import Tensor
import mindspore as ms
import mindspore.nn as nn
from mindspore.nn import TrainOneStepCell, WithLossCell
from mindspore.common.initializer import Normal

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
@pytest.mark.parametrize('nesterov', [True, False,])
def test_sgd_optimizer(opt, nesterov):

    #download_train = Mnist(path="/data/mnist/mnist_mv_format", split="test", batch_size=32, repeat_num=1, shuffle=True, resize=32, download=False)
    #dataset_train = download_train.run()
    
    network = SimpleCNN(in_channels=1, num_classes=10) #lenet(num_classes=10, pretrained=False)
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    net_opt = create_optimizer(network.trainable_params(), opt, lr=0.01, weight_decay=1e-5, momentum=0.9, nesterov=nesterov)

    #model = ms.Model(network, loss_fn=net_loss, optimizer=net_opt, metrics={'acc'})
	
    bs = 8
    input_data = Tensor(np.ones([bs, 1, 32, 32]).astype(np.float32) * 0.01)
    label = Tensor(np.ones([bs]).astype(np.int32))

    net_with_loss = WithLossCell(network, net_loss)
    train_network = TrainOneStepCell(net_with_loss, net_opt)

    train_network.set_train()

    #_cell_graph_executor.compile(train_network, inputs, label)(input_data, label)
    begin_loss = train_network(input_data, label)
    for i in range(10):
        cur_loss = train_network(input_data, label)
        #print("loss:  ", cur_loss)
    print(f"{opt}, begin loss: {begin_loss}, end loss:  {cur_loss}")

    # check output correctness 
    assert cur_loss < begin_loss, 'Loss does NOT decrease' 


@pytest.mark.parametrize('opt', ['adam', 'adamW', 'rmsprop', 'adagrad'])
def test_adam_optimizer(opt):
     
    network = SimpleCNN(num_classes=10)
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    net_opt = create_optimizer(network.trainable_params(), opt, lr=0.01, weight_decay=1e-5)

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


if __name__ == "__main__":
    test_sgd_optimizer('sgd', True)
    
