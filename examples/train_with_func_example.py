import os
import sys
from time import time

sys.path.append('.')

import mindspore as ms
from mindspore import ops, Model

from mindcv.data import create_dataset, create_transforms, create_loader
from mindcv.models import create_model
from mindcv.loss import create_loss
from mindcv.optim import create_optimizer


def main():
    ms.set_seed(1)
    ms.set_context(mode=ms.PYNATIVE_MODE)

    # --------------------------- Prepare data -------------------------#
    # create dataset for train and val
    num_classes = 10
    num_workers = 8
    data_dir = '/data/cifar-10-batches-bin'
    download = False if os.path.exists(data_dir) else True

    dataset_train = create_dataset(name='cifar10', root=data_dir, split='train', shuffle=True, download=download,
                                   num_parallel_workers=num_workers)
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
    correct /= total
    acc = 100 * correct
    print(f"Test Accuracy: {acc:>0.2f}% \n")
    return acc

if __name__ == '__main__':
    main()
