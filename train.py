''' Model training pipeline '''
import os
import logging
import mindspore as ms
import numpy as np
from mindspore import nn, Tensor
from mindspore import FixedLossScaleManager, Model
from mindspore.communication import init, get_rank, get_group_size

from mindcv.models import create_model
from mindcv.data import create_dataset, create_transforms, create_loader
from mindcv.loss import create_loss
from mindcv.optim import create_optimizer
from mindcv.scheduler import create_scheduler
from mindcv.utils import StateMonitor, Allreduce
from config import parse_args

ms.set_seed(1)
np.random.seed(1)

logger = logging.getLogger('train')
logger.setLevel(logging.INFO)
h1 = logging.StreamHandler()
formatter1 = logging.Formatter('%(message)s',)
logger.addHandler(h1)
h1.setFormatter(formatter1)

def train(args):
    ''' main train function'''
    ms.set_context(mode=args.mode)

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

    if args.num_classes is None:
        num_classes = dataset_train.num_classes()
    else:
        num_classes = args.num_classes

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
        drop_remainder=args.drop_remainder,
        is_training=True,
        mixup=args.mixup,
        num_classes=num_classes,
        transform=transform_list,
        num_parallel_workers=args.num_parallel_workers,
    )

    if args.val_while_train:
        dataset_eval = create_dataset(
            name=args.dataset,
            root=args.data_dir,
            split=args.val_split,
            num_shards=device_num,
            shard_id=rank_id,
            num_parallel_workers=args.num_parallel_workers,
            download=args.dataset_download)

        transform_list_eval = create_transforms(
            dataset_name=args.dataset,
            is_training=False,
            image_resize=args.image_resize,
            crop_pct=args.crop_pct,
            interpolation=args.interpolation,
            mean=args.mean,
            std=args.std
        )

        loader_eval = create_loader(
            dataset=dataset_eval,
            batch_size=args.batch_size,
            drop_remainder=False,
            is_training=False,
            transform=transform_list_eval,
            num_parallel_workers=args.num_parallel_workers,
        )
        # validation dataset count
        eval_count = dataset_eval.get_dataset_size()
        if args.distribute:
            all_reduce = Allreduce()
            eval_count = all_reduce(Tensor(eval_count, ms.int32))
    else:
        loader_eval = None

    num_batches = loader_train.get_dataset_size()
    # Train dataset count
    train_count = dataset_train.get_dataset_size()
    if args.distribute:
        all_reduce = Allreduce()
        train_count = all_reduce(Tensor(train_count, ms.int32))

    # create model
    network = create_model(model_name=args.model,
                           num_classes=num_classes,
                           in_channels=args.in_channels,
                           drop_rate=args.drop_rate,
                           drop_path_rate=args.drop_path_rate,
                           pretrained=args.pretrained,
                           checkpoint_path=args.ckpt_path)
    
    num_params = sum([param.size for param in network.get_parameters()])

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
                                    decay_rate=args.decay_rate,
                                    milestones=args.multi_step_decay_milestones,
                                    num_epochs=args.epoch_size)
    
    # resume training if ckpt_path is given
    if args.ckpt_path != '' and args.resume_opt: 
        opt_ckpt_path = os.path.join(args.ckpt_save_dir, f'optim_{args.model}.ckpt')
    else:
        opt_ckpt_path = '' 

    # create optimizer
    #TODO: consistent naming opt, name, dataset_name
    optimizer = create_optimizer(network.trainable_params(),
                                 opt=args.opt,
                                 lr=lr_scheduler,
                                 weight_decay=args.weight_decay,
                                 momentum=args.momentum,
                                 nesterov=args.use_nesterov,
                                 filter_bias_and_bn=args.filter_bias_and_bn,
                                 loss_scale=args.loss_scale,
                                 checkpoint_path=opt_ckpt_path)

    # Define eval metrics.
    eval_metrics = {'Top_1_Accuracy': nn.Top1CategoricalAccuracy()}

    # init model
    if args.loss_scale > 1.0:
        loss_scale_manager = FixedLossScaleManager(loss_scale=args.loss_scale, drop_overflow_update=False)
        model = Model(network, loss_fn=loss, optimizer=optimizer, metrics=eval_metrics, amp_level=args.amp_level,
                      loss_scale_manager=loss_scale_manager)
    else:
        model = Model(network, loss_fn=loss, optimizer=optimizer, metrics=eval_metrics, amp_level=args.amp_level)

    # callback
    # save checkpoint, summary training loss
    # recorad val acc and do model selection if val dataset is availabe
    begin_epoch = 0
    if args.ckpt_path != '':
        if args.ckpt_path != '':
            begin_step = optimizer.global_step.asnumpy()[0]
            begin_epoch = args.ckpt_path.split('/')[-1].split('-')[1].split('_')[0]
            begin_epoch = int(begin_epoch)

    summary_dir = f"./{args.ckpt_save_dir}/summary"
    assert (args.ckpt_save_policy != 'top_k' or args.val_while_train == True), \
        "ckpt_save_policy is top_k, val_while_train must be True."
    state_cb = StateMonitor(model, summary_dir=summary_dir,
                            dataset_val=loader_eval,
                            val_interval=args.val_interval,
                            metric_name="Top_1_Accuracy",
                            ckpt_dir=args.ckpt_save_dir,
                            ckpt_save_interval=args.ckpt_save_interval,
                            best_ckpt_name=args.model + '_best.ckpt',
                            rank_id=rank_id,
                            device_num=device_num,
                            log_interval=args.log_interval,
                            keep_checkpoint_max=args.keep_checkpoint_max,
                            model_name=args.model,
                            last_epoch=begin_epoch,
                            ckpt_save_policy=args.ckpt_save_policy)

    callbacks = [state_cb]
    # log
    if rank_id in [None, 0]:
        logger.info(f"-" * 40)
        logger.info(f"Num devices: {device_num if device_num is not None else 1} \n"
                    f"Distributed mode: {args.distribute} \n"
                    f"Num training samples: {train_count}")
        if args.val_while_train:
            logger.info(f"Num validation samples: {eval_count}")
        logger.info(f"Num classes: {num_classes} \n"
                    f"Num batches: {num_batches} \n"
                    f"Batch size: {args.batch_size} \n"
                    f"Auto augment: {args.auto_augment} \n"
                    f"Model: {args.model} \n"
                    f"Model param: {num_params} \n"
                    f"Num epochs: {args.epoch_size} \n"
                    f"Optimizer: {args.opt} \n"
                    f"LR: {args.lr} \n"
                    f"LR Scheduler: {args.scheduler}")
        logger.info(f"-" * 40)

        if args.ckpt_path != '':
            logger.info(f"Resume training from {args.ckpt_path}, last step: {begin_step}, last epoch: {begin_epoch}")
        else:
            logger.info('Start training')

    model.train(args.epoch_size, loader_train, callbacks=callbacks, dataset_sink_mode=args.dataset_sink_mode)

if __name__ == '__main__':
    args = parse_args()
    train(args)
