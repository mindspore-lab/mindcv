import os
import sys
import argparse
from time import time

sys.path.append('.')

import mindspore as ms
import mindspore.nn as nn
from mindspore import ops, Tensor, Model
from mindspore.communication import init, get_rank, get_group_size
from mindspore.parallel._utils import _get_device_num, _get_gradients_mean

from mindcv.data import create_dataset, create_transforms, create_loader
from mindcv.models import create_model
from mindcv.loss import create_loss
from mindcv.optim import create_optimizer
from mindcv.utils import StateMonitor, Allreduce


def main():
    ms.set_seed(1)
    ms.set_context(mode=ms.PYNATIVE_MODE)

    # --------------------------- Prepare data -------------------------#
    # create dataset for train and val
    init()
    device_num = get_group_size()
    rank_id = get_rank()
    ms.set_auto_parallel_context(device_num=device_num,
                                 parallel_mode='data_parallel',
                                 gradients_mean=True)
    num_classes = 10
    num_workers = 8
    data_dir = '/data/cifar-10-batches-bin'
    download = False if os.path.exists(data_dir) else True

    dataset_train = create_dataset(name='cifar10', root=data_dir, split='train', shuffle=True, download=download,
                                   num_shards=device_num, shard_id=rank_id, num_parallel_workers=num_workers)
    dataset_test = create_dataset(name='cifar10', root=data_dir, split='test', shuffle=False, download=False,
                                  num_shards=device_num, shard_id=rank_id, num_parallel_workers=num_workers)

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
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        save_path = f'./ckpt/resnet18-{t + 1}_{num_batches}.ckpt'
        b = time()
        train_epoch(network, loader_train, loss, opt)
        print('Epoch time cost: ', time() - b)
        test_epoch(network, loader_test)
        ms.save_checkpoint(network, save_path, async_save=True)
    print("Done!")


def train_epoch(network, dataset, loss_fn, optimizer):
    # Define forward function
    def forward_fn(data, label):
        logits = network(data)
        loss = loss_fn(logits, label)
        return loss, logits

    # Get gradient function
    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
    mean = _get_gradients_mean()
    degree = _get_device_num()
    grad_reducer = nn.DistributedGradReducer(optimizer.parameters, mean, degree)
    # Define function of one-step training,
    @ms.ms_function
    def train_step_parallel(data, label):
        (loss, _), grads = grad_fn(data, label)
        grads = grad_reducer(grads)
        loss = ops.depend(loss, optimizer(grads))
        return loss

    network.set_train()
    size = dataset.get_dataset_size()
    for batch, (data, label) in enumerate(dataset.create_tuple_iterator()):
        loss = train_step_parallel(data, label)
        if batch % 100 == 0:
            loss, current = loss.asnumpy(), batch
            print(f"loss: {loss:>7f}  [{current:>3d}/{size:>3d}]")


def test_epoch(network, dataset):
    network.set_train(False)
    total, correct = 0, 0
    for data, label in dataset.create_tuple_iterator():
        pred = network(data)
        total += len(data)
        if len(label.shape) == 1:
            correct += (pred.argmax(1) == label).asnumpy().sum()
        else:  # one-hot or soft label
            correct += (pred.argmax(1) == label.argmax(1)).asnumpy().sum()
    all_reduce = Allreduce()
    correct = all_reduce(Tensor(correct, ms.float32))
    total = all_reduce(Tensor(total, ms.float32))
    correct /= total
    acc = 100 * correct.asnumpy()
    print(f"Test Accuracy: {acc:>0.2f}% \n")
    return acc

if __name__ == '__main__':
    main()
