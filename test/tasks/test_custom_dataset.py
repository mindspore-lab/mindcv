#import sys
#sys.path.append('.')

import mindspore as ms
from mindcv.data import create_dataset, create_transforms, create_loader
from mindcv.models import create_model
from mindcv.loss import create_loss
from mindcv.optim import create_optimizer
from mindcv.scheduler import create_scheduler
from mindspore import FixedLossScaleManager, Model, LossMonitor, TimeMonitor, CheckpointConfig, ModelCheckpoint
import mindspore.nn as nn
import os

num_workers = 8
num_classes = 2

def test_finetune():
    ms.set_seed(1)
    #ms.context.set_context(mode=ms.context.GRAPH_MODE)
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target='GPU')

    num_epochs = 5

    # prepare dataset
    # create dataset
    # download this dataset from https://download.pytorch.org/tutorial/hymenoptera_data.zip
    data_dir = '/data/ant_and_bee'

    dataset_train = create_dataset(root=data_dir, split='train', shuffle=True, num_parallel_workers=num_workers, download=False)

    # create transform and get trans list
    trans = create_transforms(dataset_name='', is_training=True)

    # get data loader for training
    loader_train = create_loader(
            dataset=dataset_train,
            batch_size=64,
            is_training=True,
            num_classes=num_classes,
            transform=trans,
            num_parallel_workers=num_workers,
            drop_remainder=True
        )


    # build network and train
    # build resnet model
    network = create_model(model_name='densenet121', num_classes=num_classes, pretrained=True)

    # set loss function
    loss = create_loss(name='CE')

    # set optimizer 
    steps_per_epoch = loader_train.get_dataset_size()
    print('Steps per epoch: ', steps_per_epoch)

    opt = create_optimizer(network.trainable_params(), opt='adam', lr=1e-4) 

    # TODO: simplify the training code 
    ckpt_save_dir = './ckpt_custom'
    
    model = Model(network, loss_fn=loss, optimizer=opt, metrics={'accuracy'})

    loss_cb = LossMonitor(per_print_times=10)
    time_cb = TimeMonitor(data_size=10)
    callbacks = [loss_cb, time_cb]
    ckpt_config = CheckpointConfig(save_checkpoint_steps=steps_per_epoch)
    ckpt_cb = ModelCheckpoint(prefix='densenet121',
                              directory=ckpt_save_dir,
                              config=ckpt_config)
    callbacks.append(ckpt_cb)

    # train model
    set_sink_mode = True # not args.device_target == "CPU"
    model.train(num_epochs, loader_train, callbacks=callbacks, dataset_sink_mode=set_sink_mode)

    # evalate accuracy
    dataset_train = create_dataset(root=data_dir, split='val', shuffle=False, num_parallel_workers=num_workers, download=False)

    # create transform and get trans list
    trans = create_transforms(dataset_name='', is_training=False)

    # get data loader for training
    loader_eval = create_loader(
            dataset=dataset_train,
            batch_size=64,
            is_training=False,
            num_classes=num_classes,
            transform=trans,
            num_parallel_workers=num_workers,
            drop_remainder=False
        )
        
    # model 
    #network = create_model(model_name='densenet121', num_classes=num_classes, 
    #        checkpoint_path='ckpt_densenet/densenet121-3_781.ckpt')
    
    #loss = create_loss(name='CE')

    # init model
    #model = Model(network, loss_fn=loss, metrics={"accuracy"})

    acc = model.eval(loader_eval)
    print('acc', acc)

def evaluate():
    # evalate accuracy
    dataset_train = create_dataset(name='cifar10', root=cifar10_dir, split='test', shuffle=False, num_parallel_workers=num_workers, download=False)

    # create transform and get trans list
    trans = create_transforms(dataset_name='cifar10')

    # get data loader for training
    loader_eval = create_loader(
            dataset=dataset_train,
            batch_size=64,
            is_training=False,
            num_classes=num_classes,
            transform=trans,
            num_parallel_workers=num_workers,
            drop_remainder=False
        )
        
    # model 
    network = create_model(model_name='densenet121', num_classes=num_classes, 
            checkpoint_path='ckpt_densenet/densenet121-3_781.ckpt')
    
    loss = create_loss(name='CE')

    # init model
    model = Model(network, loss_fn=loss, metrics={"accuracy"})

    acc = model.eval(loader_eval)
    print('acc', acc)


if __name__=='__main__':
    test_finetune()
    #evaluate()

