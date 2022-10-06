import os
from time import time

import mindspore as ms
import mindspore.ops as ops
from mindspore.communication import init, get_rank, get_group_size

from mindcv.models import create_model
from mindcv.data import create_dataset, create_transforms, create_loader
from mindcv.loss import create_loss
from mindcv.optim import create_optimizer
from mindcv.scheduler import create_scheduler
from config import parse_args


def train_epoch(network, dataset, loss_fn, optimizer, epoch, n_epochs, log_interval=100):
    # Define forward function
    def forward_fn(data, label):
        logits = network(data)
        loss = loss_fn(logits, label)
        return loss, logits

    # Get gradient function
    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    # Define function of one-step training
    @ms.ms_function
    def train_step(data, label):
        (loss, _), grads = grad_fn(data, label)
        loss = ops.depend(loss, optimizer(grads))
        return loss

    network.set_train()
    n_batches = dataset.get_dataset_size()
    n_steps = n_batches * n_epochs
    epoch_width, batch_width, step_width = len(str(n_epochs)), len(str(n_batches)), len(str(n_steps))
    step_time = time()
    for batch, (data, label) in enumerate(dataset.create_tuple_iterator()):
        loss = train_step(data, label)
        if (batch + 1) % log_interval == 0:
            step = epoch * n_batches + batch
            print(f"Epoch:[{epoch:{epoch_width}d}/{n_epochs:{epoch_width}d}], "
                  f"batch:[{batch:{batch_width}d}/{n_batches:{batch_width}d}], "
                  f"step:[{step:{step_width}d}/{n_steps:{step_width}d}], "
                  f"loss:{loss.asnumpy():8.6f}, time:{time() - step_time:.6f}s")
            step_time = time()


def train(args):
    ms.set_seed(1)
    ms.set_context(mode=ms.PYNATIVE_MODE)

    if args.distribute:
        init()
        device_num = get_group_size()
        rank_id = get_rank()
        ms.set_auto_parallel_context(device_num=device_num,
                                     parallel_mode='data_parallel',
                                     gradients_mean=True)
    else:
        device_num = None
        rank_id = None

    # create dataset
    dataset_train = create_dataset(
        name=args.dataset,
        root=args.data_dir,
        split=args.train_split,
        shuffle=args.shuffle,
        num_samples=args.num_samples,
        num_shards=device_num,
        shard_id=rank_id,
        num_parallel_workers=args.num_parallel_workers,
        download=args.dataset_download)

    # create transforms
    transform_list = create_transforms(
        dataset_name=args.dataset,
        is_training=True,
        image_resize=args.image_resize,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        interpolation=args.interpolation,
        auto_augment=args.auto_augment,
        mean=args.mean,
        std=args.std,
        re_prob=args.re_prob,
        re_scale=args.re_scale,
        re_ratio=args.re_ratio,
        re_value=args.re_value,
        re_max_attempts=args.re_max_attempts
    )

    # load dataset
    loader_train = create_loader(
        dataset=dataset_train,
        batch_size=args.batch_size,
        drop_remainder=False,
        is_training=True,
        mixup=args.mixup,
        num_classes=args.num_classes,
        transform=transform_list,
        num_parallel_workers=args.num_parallel_workers,
    )
    num_batches = loader_train.get_dataset_size()

    # create model
    network = create_model(model_name=args.model,
                           num_classes=args.num_classes,
                           in_channels=args.in_channels,
                           drop_rate=args.drop_rate,
                           drop_path_rate=args.drop_path_rate,
                           pretrained=args.pretrained,
                           checkpoint_path=args.ckpt_path)

    # create loss
    loss = create_loss(name=args.loss,
                       reduction=args.reduction,
                       label_smoothing=args.label_smoothing,
                       aux_factor=args.aux_factor)

    # create learning rate schedule
    lr_scheduler = create_scheduler(num_batches,
                                    scheduler=args.scheduler,
                                    lr=args.lr,
                                    min_lr=args.min_lr,
                                    warmup_epochs=args.warmup_epochs,
                                    decay_epochs=args.decay_epochs,
                                    decay_rate=args.decay_rate)

    # create optimizer
    optimizer = create_optimizer(network.trainable_params(),
                                 opt=args.opt,
                                 lr=lr_scheduler,
                                 weight_decay=args.weight_decay,
                                 momentum=args.momentum,
                                 nesterov=args.use_nesterov,
                                 filter_bias_and_bn=args.filter_bias_and_bn,
                                 loss_scale=args.loss_scale)

    # training
    # TODO: args.loss_scale is not making effect.
    print('Training...')
    epoch_time = time()
    for t in range(args.epoch_size):
        train_epoch(network, loader_train, loss, optimizer, epoch=t, n_epochs=args.epoch_size, log_interval=num_batches)
        print(f'Epoch {t + 1} training time: {time() - epoch_time:.3f}s')

        # Save checkpoint
        if ((t + 1) % args.ckpt_save_interval == 0) or (t+1 == args.epoch_size):
            if (not args.distribute) or (args.distribute and rank_id == 0):
                os.makedirs(args.ckpt_save_dir, exist_ok=True)
                save_path = os.path.join(args.ckpt_save_dir, f"{args.model}-{t + 1}_{num_batches}.ckpt") # for consistency with train.py 
                ms.save_checkpoint(network, save_path, async_save=True)
                print(f"Saving model to {save_path}")
        epoch_time = time()
    print("Done!")


if __name__ == '__main__':
    args = parse_args()
    train(args)
