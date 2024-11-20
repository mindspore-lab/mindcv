""" Model training pipeline """
import logging
import os

import mindspore as ms
from mindspore import Tensor
from mindspore.communication import get_group_size, get_rank, init

from mindcv.data import create_dataset, create_loader, create_transforms
from mindcv.loss import create_loss
from mindcv.models import create_model
from mindcv.optim import create_optimizer
from mindcv.scheduler import create_scheduler
from mindcv.utils import (
    AllReduceSum,
    StateMonitor,
    create_trainer,
    get_metrics,
    require_customized_train_step,
    set_logger,
    set_seed,
)

from config import parse_args, save_args  # isort: skip

logger = logging.getLogger("mindcv.train")


def finetune_train(args):
    """main train function"""

    ms.set_context(mode=args.mode)
    if args.mode == ms.GRAPH_MODE:
        ms.set_context(jit_config={"jit_level": "O2"})
    if args.distribute:
        init()
        device_num = get_group_size()
        rank_id = get_rank()
        ms.set_auto_parallel_context(
            device_num=device_num,
            parallel_mode="data_parallel",
            gradients_mean=True,
            # we should but cannot set parameter_broadcast=True, which will cause error on gpu.
        )
    else:
        device_num = None
        rank_id = None

    set_seed(args.seed)
    set_logger(name="mindcv", output_dir=args.ckpt_save_dir, rank=0 if not rank_id else rank_id, color=False)
    logger.info(
        "We recommend installing `termcolor` via `pip install termcolor` "
        "and setup logger by `set_logger(..., color=True)`"
    )

    # check directory structure of datatset (only use for offline way of reading dataset)
    # for data_split in [args.train_split, args.val_split]:
    #     path = [i for i in os.listdir(args.data_dir + "/" + data_split + "/") if
    #             os.path.isdir(args.data_dir + "/" + data_split + "/" + i + "/")]
    #     file_num = len(path)
    #     if file_num != args.num_classes:
    #         raise ValueError("The directory structure of the custom dataset should be the same as ImageNet, "
    #                          "which is, the hierarchy of root -> split -> class -> image. \n "
    #                          "Please check your directory structure of dataset.")

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
        download=args.dataset_download,
        num_aug_repeats=args.aug_repeats,
    )

    if args.num_classes is None:
        num_classes = dataset_train.num_classes()
    else:
        num_classes = args.num_classes

    # create transforms
    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits == 3, "Currently, only support 3 splits of augmentation"
        assert args.auto_augment is not None, "aug_splits should be set with one auto_augment"
        num_aug_splits = args.aug_splits

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
        re_max_attempts=args.re_max_attempts,
        separate=num_aug_splits > 0,
    )

    # load dataset
    loader_train = create_loader(
        dataset=dataset_train,
        batch_size=args.batch_size,
        drop_remainder=args.drop_remainder,
        is_training=True,
        mixup=args.mixup,
        cutmix=args.cutmix,
        cutmix_prob=args.cutmix_prob,
        num_classes=num_classes,
        transform=transform_list,
        num_parallel_workers=args.num_parallel_workers,
        separate=num_aug_splits > 0,
    )

    if args.val_while_train:
        dataset_eval = create_dataset(
            name=args.dataset,
            root=args.data_dir,
            split=args.val_split,
            num_shards=device_num,
            shard_id=rank_id,
            num_parallel_workers=args.num_parallel_workers,
            download=args.dataset_download,
        )

        transform_list_eval = create_transforms(
            dataset_name=args.dataset,
            is_training=False,
            image_resize=args.image_resize,
            crop_pct=args.crop_pct,
            interpolation=args.interpolation,
            mean=args.mean,
            std=args.std,
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
            all_reduce = AllReduceSum()
            eval_count = all_reduce(Tensor(eval_count, ms.int32))
    else:
        loader_eval = None
        eval_count = None

    num_batches = loader_train.get_dataset_size()
    # Train dataset count
    train_count = dataset_train.get_dataset_size()
    if args.distribute:
        all_reduce = AllReduceSum()
        train_count = all_reduce(Tensor(train_count, ms.int32))

    # create model
    network = create_model(
        model_name=args.model,
        num_classes=num_classes,
        in_channels=args.in_channels,
        drop_rate=args.drop_rate,
        drop_path_rate=args.drop_path_rate,
        pretrained=args.pretrained,
        checkpoint_path=args.ckpt_path,
        ema=args.ema,
    )

    num_params = sum([param.size for param in network.get_parameters()])

    # # if you want to freeze all the feature network:
    # from mindcv.models.registry import _model_pretrained_cfgs
    # # number of parameters to be updated
    # num_params = 2
    #
    # # read names of parameters in FC layer
    # classifier_names = [_model_pretrained_cfgs[args.model]["classifier"] + ".weight",
    #                     _model_pretrained_cfgs[args.model]["classifier"] + ".bias"]
    #
    # # prevent parameters in network(except the classifier) from updating
    # for param in network.trainable_params():
    #     if param.name not in classifier_names:
    #         param.requires_grad = False
    #
    #
    # # if you only want to freeze part of the network (for example, first 7 layers):
    # # read names of network layers
    # freeze_layer = ["features." + str(i) for i in range(7)]
    #
    # # prevent parameters in first 7 layers of network from updating
    # for param in network.trainable_params():
    #     for layer in freeze_layer:
    #         if layer in param.name:
    #             param.requires_grad = False

    # create loss
    loss = create_loss(
        name=args.loss,
        reduction=args.reduction,
        label_smoothing=args.label_smoothing,
        aux_factor=args.aux_factor,
    )

    # create learning rate schedule
    lr_scheduler = create_scheduler(
        num_batches,
        scheduler=args.scheduler,
        lr=args.lr,
        min_lr=args.min_lr,
        warmup_epochs=args.warmup_epochs,
        warmup_factor=args.warmup_factor,
        decay_epochs=args.decay_epochs,
        decay_rate=args.decay_rate,
        milestones=args.multi_step_decay_milestones,
        num_epochs=args.epoch_size,
        num_cycles=args.num_cycles,
        cycle_decay=args.cycle_decay,
        lr_epoch_stair=args.lr_epoch_stair,
    )

    # resume training if ckpt_path is given
    if args.ckpt_path != "" and args.resume_opt:
        opt_ckpt_path = os.path.join(args.ckpt_save_dir, f"optim_{args.model}.ckpt")
    else:
        opt_ckpt_path = ""

    # create optimizer
    # TODO: consistent naming opt, name, dataset_name
    if (
        args.loss_scale_type == "fixed"
        and args.drop_overflow_update is False
        and not require_customized_train_step(
            args.ema,
            args.clip_grad,
            args.gradient_accumulation_steps,
            args.amp_cast_list,
        )
    ):
        optimizer_loss_scale = args.loss_scale
    else:
        optimizer_loss_scale = 1.0

    # # set learning rate for specific layer:
    # # Note: a)the params-lr dict must contain all the parameters.
    # #       b)Also, you're recommended to set a dict with a key "order_params" to make sure the
    # #         parameters will be updated in a right order.
    # params_lr_group = [{"params": list(filter(lambda x: 'features.13' in x.name, network.trainable_params())),
    #                     "lr": [i * 1.05 for i in lr_scheduler]},
    #                    {"params": list(filter(lambda x: 'features.14' in x.name, network.trainable_params())),
    #                     "lr": [i * 1.1 for i in lr_scheduler]},
    #                    {"params": list(filter(lambda x: 'features.15' in x.name, network.trainable_params())),
    #                     "lr": [i * 1.15 for i in lr_scheduler]},
    #                    {"params": list(filter(
    #                        lambda x: ".".join(x.name.split(".")[:2]) not in ["features.13", "features.14",
    #                                                                          "features.15"],
    #                        network.trainable_params())),
    #                        "lr": lr_scheduler},
    #                    {"order_params": network.trainable_params()}]
    #
    # optimizer = create_optimizer(params_lr_group,
    #                              opt=args.opt,
    #                              lr=lr_scheduler,
    #                              ...)

    optimizer = create_optimizer(
        network.trainable_params(),
        opt=args.opt,
        lr=lr_scheduler,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        nesterov=args.use_nesterov,
        weight_decay_filter=args.weight_decay_filter,
        loss_scale=optimizer_loss_scale,
        checkpoint_path=opt_ckpt_path,
        eps=args.eps,
    )

    # Define eval metrics.
    metrics = get_metrics(num_classes)

    # create trainer
    trainer = create_trainer(
        network,
        loss,
        optimizer,
        metrics,
        amp_level=args.amp_level,
        amp_cast_list=args.amp_cast_list,
        loss_scale_type=args.loss_scale_type,
        loss_scale=args.loss_scale,
        drop_overflow_update=args.drop_overflow_update,
        ema=args.ema,
        ema_decay=args.ema_decay,
        clip_grad=args.clip_grad,
        clip_value=args.clip_value,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    # callback
    # save checkpoint, summary training loss
    # record val acc and do model selection if val dataset is available

    summary_dir = f"./{args.ckpt_save_dir}/summary"
    assert (
        args.ckpt_save_policy != "top_k" or args.val_while_train is True
    ), "ckpt_save_policy is top_k, val_while_train must be True."
    state_cb = StateMonitor(
        trainer,
        model_name=args.model,
        model_ema=args.ema,
        dataset_sink_mode=args.dataset_sink_mode,
        dataset_val=loader_eval,
        metric_name=list(metrics.keys()),
        val_interval=args.val_interval,
        ckpt_save_dir=args.ckpt_save_dir,
        ckpt_save_interval=args.ckpt_save_interval,
        ckpt_save_policy=args.ckpt_save_policy,
        ckpt_keep_max=args.keep_checkpoint_max,
        summary_dir=summary_dir,
        log_interval=args.log_interval,
        rank_id=rank_id,
        device_num=device_num,
    )

    callbacks = [state_cb]
    essential_cfg_msg = "\n".join(
        [
            "Essential Experiment Configurations:",
            f"MindSpore mode[GRAPH(0)/PYNATIVE(1)]: {args.mode}",
            f"Distributed mode: {args.distribute}",
            f"Number of devices: {device_num if device_num is not None else 1}",
            f"Number of training samples: {train_count}",
            f"Number of validation samples: {eval_count}",
            f"Number of classes: {num_classes}",
            f"Number of batches: {num_batches}",
            f"Batch size: {args.batch_size}",
            f"Auto augment: {args.auto_augment}",
            f"MixUp: {args.mixup}",
            f"CutMix: {args.cutmix}",
            f"Model: {args.model}",
            f"Model parameters: {num_params}",
            f"Number of epochs: {args.epoch_size}",
            f"Optimizer: {args.opt}",
            f"Learning rate: {args.lr}",
            f"LR Scheduler: {args.scheduler}",
            f"Momentum: {args.momentum}",
            f"Weight decay: {args.weight_decay}",
            f"Auto mixed precision: {args.amp_level}",
            f"Loss scale: {args.loss_scale}({args.loss_scale_type})",
        ]
    )
    logger.info(essential_cfg_msg)
    save_args(args, os.path.join(args.ckpt_save_dir, f"{args.model}.yaml"), rank_id)

    logger.info(f"Load checkpoint from {args.ckpt_path}. \nStart training")

    trainer.train(args.epoch_size, loader_train, callbacks=callbacks, dataset_sink_mode=args.dataset_sink_mode)


if __name__ == "__main__":
    args = parse_args()
    finetune_train(args)
