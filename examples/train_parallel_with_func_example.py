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

parser = argparse.ArgumentParser()
parser.add_argument('--dynamic', type=int, default=0)
parser.add_argument('--train_step', type=int, default=1)
args = parser.parse_args()


dynamic_mode = args.dynamic # use PyNative if True
train_step_mode = args.train_step # use train_step if True, otherwise use model.train

def main():
    ms.set_seed(1)

    if dynamic_mode:
        ms.set_context(mode=ms.PYNATIVE_MODE, device_target='GPU')
    else:
        ms.set_context(mode=ms.GRAPH_MODE, device_target='GPU')

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
    if train_step_mode:
        for t in range(epochs):
            print(f"Epoch {t + 1}\n-------------------------------")
            save_path = f'./ckpt/resnet18-{t + 1}_{num_batches}.ckpt'
            b = time()
            train_epoch(network, loader_train, loss, opt)
            print('Epoch time cost: ', time() - b)
            test_epoch(network, loader_test)
            ms.save_checkpoint(network, save_path, async_save=True)
        print("Done!")

    else:
        model = Model(network, loss_fn=loss, optimizer=opt, metrics={"Accuracy": nn.Accuracy()})
        summary_dir = f"./ckpt/summary"
        state_cb = StateMonitor(model, summary_dir=summary_dir,
                                dataset_val=loader_test,
                                metric_name="Accuracy",
                                ckpt_dir="./ckpt",
                                best_ckpt_name='resnet18_best.ckpt',
                                dataset_sink_mode=True,
                                rank_id=rank_id,
                                device_num=device_num,
                                distribute=True,
                                model_name="resnet18",
                                save_strategy='latest_K')

        callbacks = [state_cb]
        model.train(epochs, loader_train, dataset_sink_mode=True, callbacks=callbacks)

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
    def train_step_parallel_graph(data, label):
        (loss, _), grads = grad_fn(data, label)
        loss = ops.depend(loss, optimizer(grads))
        return loss

    def train_step_parallel(data, label):
        (loss, _), grads = grad_fn(data, label)
        loss = ops.depend(loss, optimizer(grads))
        return loss

    network.set_train()
    size = dataset.get_dataset_size()
    for batch, (data, label) in enumerate(dataset.create_tuple_iterator()):
        if dynamic_mode:
            loss = train_step_parallel(data, label)
        else:
            loss = train_step_parallel_graph(data, label)
        if batch % 100 == 0:
            loss, current = loss.asnumpy(), batch
            print(f"loss: {loss:>7f}  [{current:>3d}/{size:>3d}]")


def test_epoch(network, dataset):
    network.set_train(False)
    total = ms.Parameter(default_input=ms.Tensor(0, ms.float32), requires_grad=False)
    correct = ms.Parameter(default_input=ms.Tensor(0, ms.float32), requires_grad=False)
    for data, label in dataset.create_tuple_iterator():
        pred = network(data)
        total += len(data)
        correct += (pred.argmax(1) == label).sum().asnumpy()
    all_reduce = Allreduce()
    correct = all_reduce(correct)
    total = all_reduce(total)
    correct /= total
    acc = 100 * correct.asnumpy()
    print(f"Test Accuracy: {correct:>0.2f}% \n")
    return acc

if __name__ == '__main__':
    main()
