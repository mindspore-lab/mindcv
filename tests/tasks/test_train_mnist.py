import sys
sys.path.append('.')

import os
import pytest
import mindspore as ms
from mindspore import nn

from mindcv.data import create_dataset, create_transforms, create_loader
from mindcv.models import create_model
from mindcv.loss import create_loss
from mindcv.optim import create_optimizer
from mindcv.scheduler import create_scheduler
from mindspore import FixedLossScaleManager, Model, LossMonitor, TimeMonitor, CheckpointConfig, ModelCheckpoint


@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_train_mnist(mode):
    '''
    test mobilenet_v1_train_gpu(single)
    '''
    num_workers = 2
    num_classes = 10
    batch_size = 16
    num_epochs = 1

    set_sink_mode = True

    dataset_name = 'mnist'
    data_dir = './datasets/mnist'
    model_name = 'resnet18'
    scheduler_name = 'constant'
    lr = 1e-3
    loss_name = 'CE'
    opt_name = 'adam'
    #ckpt_save_dir = './tests/ckpt_tmp'
    data_dir 

    if not os.path.exists(data_dir):
        download = True
    else:
        download = False

    ms.set_seed(1)
    ms.set_context(mode=mode)

    device_num = None
    rank_id = None

    dataset_train = create_dataset(
        name=dataset_name,
        root=data_dir,
        num_samples=100,
        num_shards=device_num,
        shard_id=rank_id,
        download=download
    )

    transform_train = create_transforms(
        dataset_name=dataset_name
    )

    loader_train = create_loader(
        dataset=dataset_train,
        batch_size=batch_size,
        is_training=True,
        num_classes=num_classes,
        transform=transform_train,
        num_parallel_workers=num_workers,
        drop_remainder=True
    )

    network = create_model(
        model_name=model_name,
        in_channels=1,
        num_classes=num_classes
    )

    loss = create_loss(name=loss_name)

    net_with_criterion = nn.WithLossCell(network, loss)

    steps_per_epoch = loader_train.get_dataset_size()
    print('Steps per epoch: ', steps_per_epoch)

    lr_scheduler = create_scheduler(
        steps_per_epoch=steps_per_epoch,
        scheduler=scheduler_name,
        lr=lr
    )

    opt = create_optimizer(
        network.trainable_params(),
        opt=opt_name,
        lr=lr_scheduler
    )

    train_network = nn.TrainOneStepCell(network=net_with_criterion, optimizer=opt)
    train_network.set_train()
    losses = []
    
    num_steps = 0
    max_steps = 10
    while num_steps < max_steps:
        for batch, (data, label) in enumerate(loader_train.create_tuple_iterator()):
            loss = train_network(data, label)
            losses.append(loss)
            print(loss)

            num_steps += 1

    assert losses[num_steps-1] < losses[0], 'Loss does NOT decrease'

if __name__=='__main__':
    test_train_mnist(ms.GRAPH_MODE)
