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


def main():
    args = parse_args()
    ms.set_context(mode=args.mode)
    if args.mode == ms.GRAPH_MODE:
        ms.set_context(jit_config={"jit_level": "O2"})
    if args.distribute:
        init()
        rank_id, device_num = get_rank(), get_group_size()
        ms.set_auto_parallel_context(
            device_num=device_num,
            parallel_mode="data_parallel",
            gradients_mean=True,
            # we should but cannot set parameter_broadcast=True, which will cause error on gpu.
        )
        all_reduce = AllReduceSum()
    else:
        rank_id, device_num = None, None
        all_reduce = None

    set_seed(args.seed)
    set_logger(name="mindcv", output_dir=args.ckpt_save_dir, rank=rank_id, color=False)
    logger.info(
        "We recommend installing `termcolor` via `pip install termcolor` "
        "and setup logger by `set_logger(..., color=True)`"
    )

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
    num_batches = loader_train.get_dataset_size()
    train_count = dataset_train.get_dataset_size()
    if args.distribute:
        train_count = all_reduce(Tensor(train_count, ms.int32))

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
        eval_count = dataset_eval.get_dataset_size()
        if args.distribute:
            eval_count = all_reduce(Tensor(eval_count, ms.int32))
    else:
        loader_eval = None
        eval_count = None

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
    optimizer = create_optimizer(
        network,
        opt=args.opt,
        lr=lr_scheduler,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        nesterov=args.use_nesterov,
        weight_decay_filter=args.weight_decay_filter,
        layer_decay=args.layer_decay,
        loss_scale=optimizer_loss_scale,
        checkpoint_path=opt_ckpt_path,
        eps=args.eps,
    )

    # define eval metrics.
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
    begin_step = 0
    begin_epoch = 0
    if args.ckpt_path != "":
        begin_step = optimizer.global_step.asnumpy()[0]
        begin_epoch = args.ckpt_path.split("/")[-1].split("-")[1].split("_")[0]
        begin_epoch = int(begin_epoch)

    summary_dir = f"./{args.ckpt_save_dir}/summary"
    assert (
        args.ckpt_save_policy != "top_k" or args.val_while_train is True
    ), "ckpt_save_policy is top_k, val_while_train must be True."
    state_cb = StateMonitor(
        trainer,
        model_name=args.model,
        model_ema=args.ema,
        last_epoch=begin_epoch,
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

    if args.ckpt_path != "":
        logger.info(f"Resume training from {args.ckpt_path}, last step: {begin_step}, last epoch: {begin_epoch}")
    else:
        logger.info("Start training")

    trainer.train(args.epoch_size, loader_train, callbacks=callbacks, dataset_sink_mode=args.dataset_sink_mode)


if __name__ == "__main__":
    main()
