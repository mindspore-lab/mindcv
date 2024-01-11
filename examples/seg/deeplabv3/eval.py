import argparse
import logging

import yaml
from addict import Dict
from data import create_segment_dataset
from deeplabv3 import DeepLabInferNetwork, DeepLabV3, DeepLabV3Plus
from dilated_resnet import *  # noqa: F403, F401
from postprocess import apply_eval

from mindspore import load_checkpoint, load_param_into_net

from mindcv.models import create_model

logger = logging.getLogger("deeplabv3.eval")


def eval(args):
    # create eval dataset
    eval_dataset = create_segment_dataset(
        name=args.dataset,
        data_dir=args.eval_data_lst,
        is_training=False,
    )

    # create eval model
    backbone = create_model(
        args.backbone,
        features_only=args.backbone_features_only,
        out_indices=args.backbone_out_indices,
        output_stride=args.output_stride,
    )

    if args.model == "deeplabv3":
        deeplab = DeepLabV3(backbone, args, is_training=False)
    elif args.model == "deeplabv3plus":
        deeplab = DeepLabV3Plus(backbone, args, is_training=False)
    else:
        NotImplementedError("support deeplabv3 and deeplabv3plus only")

    eval_model = DeepLabInferNetwork(deeplab, input_format=args.input_format)

    param_dict = load_checkpoint(args.ckpt_path)
    net_param_not_load, _ = load_param_into_net(eval_model, param_dict)
    if len(net_param_not_load) == 0:
        logger.info(f"ckpt - {args.ckpt_path} loaded successfully")

    eval_model.set_train(False)

    logger.info("\n========================================\n")
    logger.info("Processing, please wait a moment.")

    # evaluate
    eval_param_dict = {"net": eval_model, "dataset": eval_dataset, "args": args}

    mIoU = apply_eval(eval_param_dict)

    logger.info("\n========================================\n")
    logger.info(f"mean IoU: {mIoU}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation Config", add_help=False)
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

    # core evaluation
    eval(args)
