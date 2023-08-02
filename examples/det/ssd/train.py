import argparse
import os
import sys

import yaml
from addict import Dict
from callbacks import get_ssd_callbacks, get_ssd_eval_callback
from data import create_ssd_dataset
from model import SSD, SSDInferWithDecoder, SSDWithLossCell, get_ssd_trainer
from utils import get_ssd_lr_scheduler, get_ssd_optimizer

import mindspore as ms
from mindspore.communication import get_group_size, get_rank, init

sys.path.append(".")

from mindcv.models import create_model
from mindcv.utils import set_seed


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

    dataset = create_ssd_dataset(
        name=args.dataset,
        root=args.data_dir,
        shuffle=args.shuffle,
        batch_size=args.batch_size,
        python_multiprocessing=True,
        num_parallel_workers=args.num_parallel_workers,
        drop_remainder=args.drop_remainder,
        args=args,
        num_shards=device_num,
        shard_id=rank_id,
        is_training=True,
    )

    steps_per_epoch = dataset.get_dataset_size()

    # use mindcv models as backbone for SSD
    backbone = create_model(
        args.backbone,
        checkpoint_path=args.backbone_ckpt_path,
        auto_mapping=args.get("backbone_ckpt_auto_mapping", False),
        features_only=args.backbone_features_only,
        out_indices=args.backbone_out_indices,
    )

    ssd = SSD(backbone, args)
    ms.amp.auto_mixed_precision(ssd, amp_level=args.amp_level)
    model = SSDWithLossCell(ssd, args)

    lr_scheduler = get_ssd_lr_scheduler(args, steps_per_epoch)
    optimizer = get_ssd_optimizer(model, lr_scheduler, args)

    trainer = get_ssd_trainer(model, optimizer, args)

    callbacks = get_ssd_callbacks(args, steps_per_epoch, rank_id)

    if args.eval_while_train and rank_id == 0:
        eval_model = SSDInferWithDecoder(ssd, args)
        eval_dataset = create_ssd_dataset(
            name=args.dataset,
            root=args.data_dir,
            shuffle=False,
            batch_size=args.batch_size,
            python_multiprocessing=True,
            num_parallel_workers=args.num_parallel_workers,
            drop_remainder=False,
            args=args,
            is_training=False,
        )
        eval_callback = get_ssd_eval_callback(eval_model, eval_dataset, args)
        callbacks.append(eval_callback)

    trainer.train(args.epoch_size, dataset, callbacks=callbacks, dataset_sink_mode=args.dataset_sink_mode)


def parse_args():
    parser = argparse.ArgumentParser(description="Training Config", add_help=False)
    parser.add_argument(
        "-c", "--config", type=str, default="", help="YAML config file specifying default arguments (default=" ")"
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    yaml_fp = args.config

    with open(yaml_fp) as fp:
        args = yaml.safe_load(fp)

    args = Dict(args)

    # data sync for cloud platform if enabled
    if args.enable_modelarts:
        import moxing as mox

        args.data_dir = f"/cache/{args.data_url}"
        mox.file.copy_parallel(src_url=os.path.join(args.data_url, args.dataset), dst_url=args.data_dir)

    # core training
    train(args)

    if args.enable_modelarts:
        mox.file.copy_parallel(src_url=args.ckpt_save_dir, dst_url=args.train_url)
