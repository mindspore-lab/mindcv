import argparse
import logging

import yaml
from addict import Dict
from callbacks import get_segment_eval_callback, get_segment_train_callback
from data import create_segment_dataset
from deeplabv3 import DeepLabInferNetwork, DeepLabV3, DeepLabV3Plus
from dilated_resnet import *  # noqa: F403, F401
from loss import SoftmaxCrossEntropyLoss

import mindspore as ms
from mindspore import context, load_checkpoint, load_param_into_net
from mindspore.communication import get_group_size, get_rank, init

from mindcv.models import create_model
from mindcv.optim import create_optimizer
from mindcv.scheduler import create_scheduler
from mindcv.utils import create_trainer, require_customized_train_step, set_seed

logger = logging.getLogger("deeplabv3.train")


def train(args):
    """main train function"""
    ms.set_context(mode=args.mode)

    if args.distribute:
        init()
        device_num = get_group_size()
        rank_id = get_rank()
        ms.set_auto_parallel_context(
            device_num=device_num,
            parallel_mode="data_parallel",
            gradients_mean=True,
            all_reduce_fusion_config=args.all_reduce_fusion_config,
        )
    else:
        device_num = None
        rank_id = None

    set_seed(args.seed)

    # dataset
    dataset = create_segment_dataset(
        name=args.dataset,
        data_dir=args.data_dir,
        args=args,
        is_training=True,
        shard_id=rank_id,
        shard_num=device_num,
    )

    steps_per_epoch = dataset.get_dataset_size()

    # use mindcv models as backbone
    backbone = create_model(
        args.backbone,
        checkpoint_path=args.backbone_ckpt_path,
        auto_mapping=args.get("backbone_ckpt_auto_mapping", False),
        features_only=args.backbone_features_only,
        out_indices=args.backbone_out_indices,
        output_stride=args.output_stride,
    )

    # network
    if args.model == "deeplabv3":
        deeplab = DeepLabV3(backbone, args, is_training=True)
    elif args.model == "deeplabv3plus":
        deeplab = DeepLabV3Plus(backbone, args, is_training=True)
    else:
        NotImplementedError("support deeplabv3 and deeplabv3plus only")
    ms.amp.auto_mixed_precision(deeplab, amp_level=args.amp_level)

    # load pretrained ckpt
    if args.ckpt_pre_trained:
        param_dict = load_checkpoint(args.ckpt_path)
        net_param_not_load, _ = load_param_into_net(deeplab, param_dict)
        if len(net_param_not_load) == 0:
            logger.info(f"pretrained ckpt - {args.ckpt_path} loaded successfully")
        else:
            raise ValueError("inconsistent params: please check net params and ckpt params")

    # loss
    loss = SoftmaxCrossEntropyLoss(args.num_classes, args.ignore_label)

    # learning rate schedule
    lr_scheduler = create_scheduler(
        steps_per_epoch,
        scheduler=args.scheduler,
        lr=args.lr,
        min_lr=args.min_lr,
        decay_epochs=args.decay_epochs,
        decay_rate=args.decay_rate,
        milestones=args.multi_step_decay_milestones,
        num_epochs=args.epoch_size,
        lr_epoch_stair=args.lr_epoch_stair,
    )

    # create optimizer
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
        deeplab.trainable_params(),
        opt="momentum",
        lr=lr_scheduler,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        filter_bias_and_bn=args.filter_bias_and_bn,
        loss_scale=optimizer_loss_scale,
    )

    trainer = create_trainer(
        deeplab,
        loss,
        optimizer,
        metrics=None,
        amp_level="O0" if args.device_target == "CPU" else args.amp_level,
        amp_cast_list=args.amp_cast_list,
        loss_scale_type=args.loss_scale_type,
        loss_scale=args.loss_scale,
        drop_overflow_update=args.drop_overflow_update,
    )

    # callback
    callbacks = get_segment_train_callback(args, steps_per_epoch, rank_id)

    # eval when train
    if args.eval_while_train and rank_id == 0:
        eval_model = DeepLabInferNetwork(deeplab, input_format=args.input_format)
        eval_dataset = create_segment_dataset(
            name=args.dataset,
            data_dir=args.eval_data_lst,
            is_training=False,
        )
        eval_callback = get_segment_eval_callback(eval_model, eval_dataset, args)
        callbacks.append(eval_callback)

    logger.info("Start training")

    trainer.train(
        args.epoch_size,
        dataset,
        callbacks=callbacks,
        dataset_sink_mode=(args.device_target != "CPU"),
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Training Config", add_help=False)
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="",
        help="YAML config file specifying default arguments (default=" ")",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    yaml_fp = args.config
    with open(yaml_fp) as fp:
        args = yaml.safe_load(fp)
    args = Dict(args)
    context.set_context(device_target=args.device_target)

    # core training
    train(args)
