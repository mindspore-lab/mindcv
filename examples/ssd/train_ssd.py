from addict import Dict
import argparse
import logging
import os
import sys
import yaml

import mindspore as ms
from mindspore.communication import get_group_size, get_rank, init

from data import create_ssd_dataset

sys.path.append("../..")

from mindcv.models import create_model
from mindcv.utils import set_seed

# TODO: arg parser already has a logger
logger = logging.getLogger("train")
logger.setLevel(logging.INFO)
h1 = logging.StreamHandler()
formatter1 = logging.Formatter("%(message)s")
logger.addHandler(h1)
h1.setFormatter(formatter1)


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
            # we should but cannot set parameter_broadcast=True, which will cause error on gpu.
        )
    else:
        device_num = None
        rank_id = None

    set_seed(args.seed)

    dataset = create_ssd_dataset(
        args,
        num_shards=device_num,
        shard_id=rank_id,
        is_training=True,
    )

    dataset_size = dataset.get_dataset_size()

    # use mindcv models as backbone for SSD
    backbone = create_model(
        args.backbone,
        checkpoint_path=args.backbone_ckpt_path,
        auto_mapping=args.get("backbone_ckpt_auto_mapping", False),
        features_only=args.backbone_features_only,
    )

    # ssd = SSD(backbone=backbone, args)


def parse_args():
    parser = argparse.ArgumentParser(description='Training Config',
                                     add_help=False)
    parser.add_argument(
        '-c', '--config', type=str, default='',
        help='YAML config file specifying default arguments (default='')'
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
