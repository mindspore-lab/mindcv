import sys
sys.path.append('.')

import mindspore as ms
from mindcv.data import create_dataset, create_transforms, create_loader
from mindspore.communication import init, get_rank, get_group_size
from mindcv.models import create_model
from mindcv.loss import create_loss
from mindcv.optim import create_optimizer
from mindcv.scheduler import create_scheduler
from mindspore import FixedLossScaleManager, Model, LossMonitor, TimeMonitor, CheckpointConfig, ModelCheckpoint
import mindspore.nn as nn
from mindspore.nn import TrainOneStepCell, WithLossCell
from mindspore.nn.transformer import Transformer
import os
from mindspore.profiler import Profiler
import time 
from mindspore import ops
from mindvision.engine.callback import ValAccMonitor
from time import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dynamic', type=int, default=1)
parser.add_argument('--train_step', type=int, default=1)
args = parser.parse_args()


dynamic_mode = args.dynamic # use PyNative if True
train_step_mode  = args.train_step # use train_step if True, otherwise use model.train 


def train_epoch(network, dataset, loss_fn, optimizer):
    # Define forward function
    def forward_fn(data, label):
        logits = network(data)
        loss = loss_fn(logits, label)
        return loss, logits

    # Get gradient function
    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    # Define function of one-step training,
    @ms.ms_function
    def train_step(data, label):
        (loss, _), grads = grad_fn(data, label)
        loss = ops.depend(loss, optimizer(grads))
        return loss

    network.set_train()
    size = dataset.get_dataset_size()
    for batch, (data, label) in enumerate(dataset.create_tuple_iterator()):
        loss = train_step(data, label)
        if batch % 100 == 0:
            loss, current = loss.asnumpy(), batch
            print(f"loss: {loss:>7f}  [{current:>3d}/{size:>3d}]")


def test_epoch(network, dataset, loss_fn):
    num_batches = dataset.get_dataset_size()
    network.set_train(False)
    total, test_loss, correct = 0, 0, 0
    for data, label in dataset.create_tuple_iterator():
        pred = network(data)
        total += len(data)
        test_loss += loss_fn(pred, label).asnumpy()
        correct += (pred.argmax(1) == label).sum().asnumpy()
    test_loss /= num_batches
    correct /= total
    acc = correct
    print(f"Test Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return acc 

def main():
    
    ms.set_seed(1)
    
    if dynamic_mode:
        ms.set_context(mode=ms.PYNATIVE_MODE, device_target='GPU')
    else:
        ms.set_context(mode=ms.GRAPH_MODE, device_target='GPU')
    
    # --------------------------- Prepare data -------------------------#
    # create dataset for train and val
    num_classes = 10 
    num_workers = 8 
    data_dir = '/data/cifar/cifar-10-batches-bin'
    download = False if os.path.exists(data_dir) else True
        
    dataset_train = create_dataset(name='cifar10', root=data_dir, split='train', shuffle=True, num_samples=num_samples, download=download, num_parallel_workers=num_workers)
    dataset_test = create_dataset(name='cifar10', root=data_dir, split='test', shuffle=False, download=False)

    # create transform and get trans list
    trans_train = create_transforms(dataset_name='cifar10', is_training=True)
    trans_test = create_transforms(dataset_name='cifar10', is_training=False)

    # get data loader
    loader_train = create_loader(
            dataset=dataset_train,
            batch_size=64,
            is_training=True,
            num_classes=num_classes,
            transform=trans_train, 
            num_parallel_workers=num_workers,
            drop_remainder=True
            )
    loader_test = create_loader(
            dataset=dataset_test,
            batch_size=32,
            is_training=False,
            num_classes=num_classes,
            transform=trans_test)
    
    num_batches = loader_train.get_dataset_size()
    print('Num batches: ', num_batches)

    # --------------------------- Build model -------------------------#
    network = create_model(model_name='resnet18', num_classes=num_classes, pretrained=False)

    loss = create_loss(name='CE')

    opt = create_optimizer(network.trainable_params(), opt='adam', lr=1e-3) 

    # --------------------------- Training and monitoring -------------------------#
    epochs = 10
    if train_step_mode:
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            b = time()
            train_epoch(network, loader_train, loss, opt)
            print('Epoch time cost: ', time()-b)
        print("Done!")

        acc = test_epoch(network, loader_test, loss)
    else:
        model = Model(network, loss_fn=loss, optimizer=opt, metrics={"Accuracy": nn.Accuracy()} )
        #model.train(epochs, loader_train, callbacks=[ValAccMonitor(model, ds_val, num_epochs)])
        model.train(epochs, loader_train, dataset_sink_mode=True, 
                callbacks=[ms.LossMonitor(100), ms.TimeMonitor(100)])

        model.eval()

    #network_with_loss = WithLossCell(network, loss)
    #train_network = TrainOneStepCell(network_with_loss, opt)

if __name__=='__main__':
    main()

