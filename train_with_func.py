"""Use the Mixed(PyNative+jit) mode to train the network"""
import logging
import os
from time import time

import numpy as np

import mindspore as ms
from mindspore import SummaryRecord, Tensor, nn, ops
from mindspore.communication import get_group_size, get_rank, init

from mindcv.data import create_dataset, create_loader, create_transforms
from mindcv.loss import create_loss
from mindcv.models import create_model
from mindcv.optim import create_optimizer
from mindcv.scheduler import create_scheduler
from mindcv.utils import AllReduceSum, CheckpointManager, get_metrics, set_logger, set_seed

from config import parse_args, save_args  # isort: skip

try:
    from mindspore import jit
except ImportError:
    from mindspore import ms_function as jit

logger = logging.getLogger("mindcv.train_with_func")


def check_args(args):
    if args.mode == ms.GRAPH_MODE:
        logger.warning("Mode of MindSpore has to be PYNATIVE(1)! Reset `args.mode` to `ms.PYNATIVE_MODE`.")
        args.mode = ms.PYNATIVE_MODE
    if args.dataset_sink_mode:
        logger.warning("Data sink is not yet supported! Reset `args.dataset_sink_mode` to `False`.")
        args.dataset_sink_mode = False
    if args.ckpt_path != "" or args.resume_opt:
        logger.warning(
            "Resuming train is not yet supported! Reset `args.ckpt_path` to empty and `args.resume_opt` to False."
        )
        args.ckpt_path = ""
        args.resume_opt = False
    if args.amp_cast_list is not None:
        logger.warning("Customized amp list is not yet supported! Reset `args.amp_cast_list` to `None`.")
        args.amp_cast_list = None
    if args.ema:
        logger.warning("EMA is not yet supported! Reset `args.ema` to `False`.")
        args.ema = False
    if args.clip_grad:
        logger.warning("Gradient clipping is not yet supported! Reset `args.clip_grad` to `False`.")
        args.clip_grad = False
    if args.gradient_accumulation_steps != 1:
        logger.warning("Gradient accumulation is not yet supported! Reset `args.gradient_accumulation_steps` to `1`.")
        args.gradient_accumulation_steps = 1
    return args


def main():
    args = parse_args()
    args = check_args(args)
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
    )
    num_params = sum([param.size for param in network.get_parameters()])
    ms.amp.auto_mixed_precision(network, amp_level=args.amp_level)
    # todo: support customized amp list
    # todo: amp EMA model

    # create loss
    criterion = create_loss(
        name=args.loss,
        reduction=args.reduction,
        label_smoothing=args.label_smoothing,
        aux_factor=args.aux_factor,
    )
    criterion = criterion.to_float(ms.float32)  # keep loss in fp32, it will automatically cast input to fp32

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

    # create optimizer
    optimizer = create_optimizer(
        network.trainable_params(),
        opt=args.opt,
        lr=lr_scheduler,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        nesterov=args.use_nesterov,
        weight_decay_filter=args.weight_decay_filter,
        loss_scale=1.0,
        eps=args.eps,
    )

    # define eval metrics.
    metrics = get_metrics(num_classes)

    # build train_step
    train_step = build_train_step(
        network,
        criterion,
        optimizer,
        loss_scale_type=args.loss_scale_type,
        loss_scale=args.loss_scale,
        drop_overflow_update=args.drop_overflow_update,
        distribute=args.distribute,
    )

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
    logger.info("Start training")

    # Training
    ckpt_settings = dict(
        name=args.model,
        path=args.ckpt_save_dir,
        interval=args.ckpt_save_interval,
        policy=args.ckpt_save_policy,
        keep=args.keep_checkpoint_max,
    )
    with SummaryRecord(f"./{args.ckpt_save_dir}/summary") as summary_record:
        # fmt: off
        train(loader_train, loader_eval, network, criterion, optimizer, train_step, metrics, args.epoch_size, args.seed,
              ckpt_settings, summary_record, args.val_interval, args.log_interval, rank_id, device_num, all_reduce)
        # fmt: on


def train(loader_train, loader_eval, network, criterion, optimizer, train_step, metrics, num_epochs, seed,
          ckpt_settings, summary_record, val_interval, log_interval, rank_id, device_num, all_reduce):  # fmt: skip
    num_batches = loader_train.get_dataset_size()
    ckpt_save_name, ckpt_save_dir, ckpt_save_interval, ckpt_save_policy, ckpt_keep_max = ckpt_settings.values()
    log_file = os.path.join(ckpt_save_dir, "result.log")
    if rank_id in [0, None]:
        os.makedirs(ckpt_save_dir, exist_ok=True)
        log_line = "".join(
            f"{s:<20}" for s in ["Epoch", "TrainLoss", *metrics.keys(), "TrainTime", "EvalTime", "TotalTime"]
        )
        with open(log_file, "w", encoding="utf-8") as fp:  # writing the title of result.log
            fp.write(log_line + "\n")
    best_acc, best_epoch = 0, -1
    need_flush_from_cache = True
    ckpt_manager = CheckpointManager(ckpt_save_policy=ckpt_save_policy)

    for epoch in range(num_epochs):
        epoch_ts = time()
        train_loss, train_acc = train_epoch(
            loader_train,
            network,
            criterion,
            optimizer,
            train_step,
            seed=seed,
            epoch=epoch,
            num_epochs=num_epochs,
            reduce_fn=all_reduce,
            summary_record=summary_record,
            log_interval=log_interval,
            rank_id=rank_id,
        )
        logger.info(f"Training accuracy: {(train_acc.asnumpy()):.4%}")
        train_time = time() - epoch_ts

        # val while train
        val_time = 0
        val_acc = np.zeros(len(metrics.keys()), dtype=np.float32)
        if loader_eval is not None and (epoch + 1) % val_interval == 0:
            val_time = time()
            val_acc = test_epoch(loader_eval, network, metrics, all_reduce, device_num)
            val_time = time() - val_time
            metric_str = "Validation "
            for metric_name, metric_value in zip(metrics.keys(), val_acc):
                metric_str += f"{metric_name}: {metric_value:.4%}, "
            metric_str += f"time: {val_time:.6f}s"
            logger.info(metric_str)
            if val_acc[0] > best_acc:
                best_acc = val_acc[0]
                best_epoch = epoch + 1
                logger.info(f"=> New best val acc: {val_acc[0]:.4%}")

        # save checkpoint
        if rank_id in [0, None]:
            if best_epoch == epoch + 1:  # always save ckpt if cur epoch got best acc
                # todo: do we need to flush_from_cache first?
                best_ckpt_save_path = os.path.join(ckpt_save_dir, f"{ckpt_save_name}-best.ckpt")
                ms.save_checkpoint(network, best_ckpt_save_path, async_save=True)
            if (epoch + 1) % ckpt_save_interval == 0 or epoch + 1 == num_epochs:
                if need_flush_from_cache:
                    need_flush_from_cache = flush_from_cache(network)
                # save optim for resume
                optim_save_path = os.path.join(ckpt_save_dir, f"{ckpt_save_name}_optim.ckpt")
                ms.save_checkpoint(optimizer, optim_save_path, async_save=True)
                ckpt_save_path = os.path.join(ckpt_save_dir, f"{ckpt_save_name}-{epoch + 1}_{num_batches}.ckpt")
                logger.info(f"Saving model to {ckpt_save_path}")
                ckpt_manager.save_ckpoint(network, num_ckpt=ckpt_keep_max, metric=val_acc[0], save_path=ckpt_save_path)

        # logging
        total_time = time() - epoch_ts
        logger.info(
            f"Total time since last epoch: {total_time:.6f}(train: {train_time:.6f}, val: {val_time:.6f})s, "
            f"ETA: {(num_epochs - epoch - 1) * total_time:.6f}s"
        )
        logger.info("-" * 80)
        if rank_id in [0, None]:
            log_line = "".join(
                f"{s:<20}"
                for s in [
                    f"{epoch + 1}",
                    f"{train_loss.asnumpy():.6f}",
                    *[f"{i:.4%}" for i in val_acc],
                    f"{train_time:.2f}",
                    f"{val_time:.2f}",
                    f"{total_time:.2f}",
                ]
            )
            with open(log_file, "a", encoding="utf-8") as fp:
                fp.write(log_line + "\n")

        # summary
        summary_record.add_value("scalar", f"train_acc_rank{rank_id}", train_acc)
        for metric_name, metric_value in zip(metrics.keys(), val_acc):
            summary_record.add_value("scalar", f"val_{metric_name}_rank{rank_id}", Tensor(metric_value))
        summary_record.record((epoch + 1) * num_batches)

    logger.info("Finish training!")
    if loader_eval is not None:
        logger.info(f"The best validation {list(metrics.keys())[0]} is: {best_acc:.4%} at epoch {best_epoch}.")
    logger.info("=" * 80)


def train_epoch(
    dataloader,
    network,
    criterion,
    optimizer,
    train_step,
    seed,
    epoch,
    num_epochs,
    reduce_fn=None,
    summary_record=None,
    log_interval=100,
    rank_id=None,
):
    network.set_train()
    criterion.set_train()
    optimizer.set_train()
    ms.dataset.config.set_seed(seed + epoch)

    num_batches = dataloader.get_dataset_size()
    epoch_width, batch_width = len(str(num_epochs)), len(str(num_batches))
    loss, correct, total = Tensor(0, ms.float32), Tensor(0, ms.float32), Tensor(0, ms.float32)
    step_ts = time()
    step_time_accum = 0
    for batch, (data, label) in enumerate(dataloader.create_tuple_iterator()):
        step = epoch * num_batches + batch
        loss, logits = train_step(data, label)
        if len(label.shape) == 1:
            correct += ops.equal(logits.argmax(1), label).sum()
        else:  # one-hot or soft label
            correct += ops.equal(logits.argmax(1), label.argmax(1)).sum()
        total += len(data)

        step_time_accum += time() - step_ts
        if (batch + 1) % log_interval == 0 or (batch + 1) == num_batches or batch == 0:
            summary_record.add_value("scalar", f"train_loss_rank{rank_id}", loss)
            summary_record.record(step + 1)
            if optimizer.dynamic_lr:
                lr = optimizer.learning_rate(Tensor(step))  # todo: this is not the real lr since may drop overflow
            else:
                lr = optimizer.learning_rate
            logger.info(
                f"Epoch: [{epoch + 1:{epoch_width}d}/{num_epochs:{epoch_width}d}], "
                f"batch: [{batch + 1:{batch_width}d}/{num_batches:{batch_width}d}], "
                f"loss: {loss.asnumpy():.6f}, "
                f"lr: {lr.asnumpy():.6f}, "
                f"time: {step_time_accum:.6f}s"
            )
            step_time_accum = 0
        step_ts = time()

    dataloader.reset()  # why do we need this?
    if reduce_fn:
        correct, total = reduce_fn(correct), reduce_fn(total)
    correct /= total

    return loss, correct


def build_train_step(
    network, criterion, optimizer, loss_scale_type, loss_scale, drop_overflow_update=True, distribute=True
):
    from mindspore.amp import DynamicLossScaler, StaticLossScaler, all_finite

    grad_reducer = nn.DistributedGradReducer(optimizer.parameters) if distribute else ops.identity
    if loss_scale_type == "fixed":
        loss_scaler = StaticLossScaler(scale_value=loss_scale)
    elif loss_scale_type == "dynamic":
        loss_scaler = DynamicLossScaler(scale_value=loss_scale, scale_factor=2, scale_window=2000)
    else:
        raise ValueError(f"Loss scale type only support ['fixed', 'dynamic'], but got{loss_scale_type}.")

    def forward_fn(data, label):
        logits = network(data)
        loss = criterion(logits, label)
        loss = loss_scaler.scale(loss)
        return loss, logits

    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    @jit
    def train_step(data, label):
        (loss, logits), grads = grad_fn(data, label)
        grads = grad_reducer(grads)
        loss = loss_scaler.unscale(loss)
        grads = loss_scaler.unscale(grads)
        status = all_finite(grads)

        if drop_overflow_update:
            if status:
                loss = ops.depend(loss, optimizer(grads))
        else:
            loss = ops.depend(loss, optimizer(grads))
        loss = ops.depend(loss, loss_scaler.adjust(status))

        # if you want to get anything about training status, return it from here and logging it outside!
        return loss, logits

    return train_step


def test_epoch(dataloader, network, metrics, reduce_fn=None, device_num=1):
    """Test network accuracy and loss."""
    network.set_train(False)

    for metric_name, metric in metrics.items():
        metric.clear()
    for data, label in dataloader.create_tuple_iterator():
        pred = network(data)
        for metric_name, metric in metrics.items():
            metric.update(pred, label)
    res_array = ms.Tensor([metric.eval() for metric_name, metric in metrics.items()], ms.float32)
    if reduce_fn:
        res_array = reduce_fn(res_array)
        res_array /= device_num
    res_array = res_array.asnumpy()

    return res_array


def flush_from_cache(network):
    """Flush cache data to host if tensor is cache enable."""
    has_cache_params = False
    params = network.get_parameters()
    for param in params:
        if param.cache_enable:
            has_cache_params = True
            Tensor(param).flush_from_cache()
    return has_cache_params


if __name__ == "__main__":
    main()
