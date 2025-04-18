""" Model training pipeline """
import logging
import os

import mindspore as ms
from mindspore import Tensor
from mindspore.communication import get_group_size, get_rank, init
from mindspore.dataset import GeneratorDataset

from mindcv.data.token_data import ImageTokenDataset
from mindcv.loss import create_loss
from mindcv.models import create_model
from mindcv.optim import create_optimizer
from mindcv.scheduler import create_scheduler
from mindcv.utils import (
    AllReduceSum,
    StateMonitor,
    create_trainer,
    get_metrics_for_token_model,
    require_customized_train_step,
    set_logger,
    set_seed,
)

from config import parse_args, save_args  # isort: skip

logger = logging.getLogger("mindcv.train")


def main():
    args = parse_args()
    ms.set_context(mode=args.mode)
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

    # TODO: move it inside
    pad_info = {
        "token": ([args.max_seq_length, 3 * args.patch_size * args.patch_size], 0),
        "pos": ([args.max_seq_length, 2], 0),
        "ind": ([args.max_seq_length], -1),
        "y": ([args.max_num_each_group], -1),  # ignore_index in loss function
    }

    num_classes = args.num_classes

    # create dataset
    dataset_train = ImageTokenDataset(
        args.data_dir,
        split=args.train_split,
        patch_size=args.patch_size,
        max_seq_length=args.max_seq_length,
        enable_cache=True,
        cache_path="train_db_info.json",
        interpolation=args.interpolation,
        image_resize=args.image_resize,
        max_num_each_group=args.max_num_each_group,
    )

    # load dataset
    loader_train = GeneratorDataset(
        dataset_train,
        column_names=["token", "pos", "ind", "y"],
        num_shards=device_num,
        shard_id=rank_id,
        num_parallel_workers=args.num_parallel_workers,
        max_rowsize=12,
        shuffle=True,
    )
    loader_train = loader_train.padded_batch(args.batch_size, drop_remainder=args.drop_remainder, pad_info=pad_info)
    num_batches = loader_train.get_dataset_size()
    train_count = len(dataset_train)
    if args.distribute:
        train_count = all_reduce(Tensor(train_count, ms.int32))

    if args.val_while_train:
        dataset_eval = ImageTokenDataset(
            args.data_dir,
            split=args.val_split,
            patch_size=args.patch_size,
            max_seq_length=args.max_seq_length,
            enable_cache=True,
            cache_path="val_db_info.json",
            interpolation=args.interpolation,
            image_resize=args.image_resize,
            max_num_each_group=args.max_num_each_group,
        )

        loader_eval = GeneratorDataset(
            dataset_eval,
            column_names=["token", "pos", "ind", "y"],
            num_shards=device_num,
            shard_id=rank_id,
            num_parallel_workers=args.num_parallel_workers,
            max_rowsize=12,
            shuffle=False,
        )
        loader_eval = loader_eval.padded_batch(args.batch_size, drop_remainder=False, pad_info=pad_info)
        eval_count = len(dataset_eval)
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
        max_num_each_group=args.max_num_each_group,
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
    metrics = get_metrics_for_token_model(num_classes)

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
        method="token",
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
        update_dataloader_groups=True,
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
