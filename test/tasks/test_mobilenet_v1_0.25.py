import os
import pytest
import mindspore as ms
import mindspore.nn as nn

from mindcv.data import create_dataset, create_transforms, create_loader
from mindcv.models import create_model
from mindcv.loss import create_loss
from mindcv.optim import create_optimizer
from mindcv.scheduler import create_scheduler
from mindspore import FixedLossScaleManager, Model, LossMonitor, TimeMonitor, CheckpointConfig, ModelCheckpoint


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard_or_eightcards
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE, ])
def test_mobilenet_v1_train_gpu(mode):
    '''
    test mobilenet_v1_train_gpu(single)
    command: mpirun -n 8 pytest -s test_mobilenet_v1_0.25.py::test_mobilenet_v1_train_gpu
    '''
    num_workers = 8
    num_classes = 10
    batch_size = 16
    num_epochs = 1

    set_sink_mode = True

    dataset_name = 'mnist'
    model_name = 'mobilenet_v1_025'
    scheduler_name = 'constant'
    lr = 1e-4
    loss_name = 'CE'
    opt_name = 'sgd'
    ckpt_save_dir = './mnist_ckpt_dir'

    if not os.path.exists(ckpt_save_dir):
        os.mkdir(ckpt_save_dir)

    ms.set_seed(1)
    ms.set_context(mode=mode)

    device_num = None
    rank_id = None

    dataset_train = create_dataset(
        name=dataset_name,
        num_samples=100,
        num_shards=device_num,
        shard_id=rank_id,
        download=True
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

    '''

    model = Model(network=network, loss_fn=loss, optimizer=opt, metrics={'accuracy'})

    loss_cb = LossMonitor(per_print_times=steps_per_epoch)
    time_cb = TimeMonitor(data_size=steps_per_epoch)
    ckpt_config = CheckpointConfig(save_checkpoint_steps=steps_per_epoch)
    ckpt_cb = ModelCheckpoint(prefix='mobile_net_v1_025_mnist',
                              directory=ckpt_save_dir,
                              config=ckpt_config)
    callbacks = [loss_cb, time_cb, ckpt_cb]

    model.train(num_epochs, loader_train, callbacks=callbacks, dataset_sink_mode=set_sink_mode)

    '''

    train_network = nn.TrainOneStepCell(network=net_with_criterion, optimizer=opt)
    train_network.set_train()
    losses = []

    dataset = dataset_train.create_dict_iterator()
    for _ in range(5):
        data = next(dataset)
        image = data["image"]
        label = data["label"]
        loss = train_network(image, label)
        losses.append(loss)
    assert losses[4] < losses[0], 'Loss does NOT decrease'


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_evaling
@pytest.mark.env_onecard
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE, ])
def test_mobilenet_v1_eval_gpu(mode):
    num_workers = 8
    num_classes = 1000
    batch_size = 64

    dataset_name = 'imagenet'
    dataset_dir = '/home/mindspore/dataset/imagenet2012/imagenet/imagenet_original'
    ckpt_dir = '/home/mindspore/junnan/mindcv-github.9.23/test/tasks/mobilenet_v1_025.ckpt'
    val_split = 'val'
    model_name = 'mobilenet_v1_025'
    loss_name = 'CE'

    ms.set_seed(1)
    ms.set_context(mode=mode)

    dataset_eval = create_dataset(
        name=dataset_name,
        root=dataset_dir,
        split=val_split,
        num_parallel_workers=num_workers
    )

    transform_eval = create_transforms(
        dataset_name=dataset_name,
        is_training=False
    )

    loader_eval = create_loader(
        dataset=dataset_eval,
        batch_size=batch_size,
        drop_remainder=False,
        is_training=False,
        transform=transform_eval,
        num_parallel_workers=num_workers,
    )

    network = create_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=True,
        checkpoint_path=ckpt_dir
    )
    network.set_train(False)

    loss = create_loss(name=loss_name)

    eval_metrics = {'Top_1_Accuracy': nn.Top1CategoricalAccuracy(),
                    'Top_5_Accuracy': nn.Top5CategoricalAccuracy()}

    model = Model(network, loss_fn=loss, metrics=eval_metrics)

    model.eval(loader_eval)
