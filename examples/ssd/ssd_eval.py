from addict import Dict
import argparse
import os
import sys
import yaml

from mindspore import load_checkpoint, load_param_into_net

from data import create_ssd_dataset
from ssd_model import SSD, SSDInferWithDecoder
from utils import apply_eval

sys.path.append("../..")

from mindcv.models import create_model


def eval(args):
    eval_dataset = create_ssd_dataset(
        name=args.dataset,
        root=args.data_dir,
        shuffle=False,
        batch_size=1,
        python_multiprocessing=False,
        num_parallel_workers=args.num_parallel_workers,
        drop_remainder=False,
        args=args,
        is_training=False
    )

    backbone = create_model(
        args.backbone,
        features_only=args.backbone_features_only,
    )

    ssd = SSD(backbone, args, is_training=False)
    eval_model = SSDInferWithDecoder(ssd, args)
    eval_model.init_parameters_data()

    param_dict = load_checkpoint(args.ckpt_path)
    load_param_into_net(eval_model, param_dict)

    eval_model.set_train(False)

    print("\n========================================\n")
    print("total images num: ", eval_dataset.get_dataset_size())
    print("Processing, please wait a moment.")

    if args.dataset == "coco":
        anno_json = os.path.join(args.data_dir, "annotations/instances_val2017.json")

    eval_param_dict = {"net": eval_model, "dataset": eval_dataset, "anno_json": anno_json, "args": args}
    mAP = apply_eval(eval_param_dict)

    print("\n========================================\n")
    print(f"mAP: {mAP}")


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

    # core evaluation
    eval(args)
