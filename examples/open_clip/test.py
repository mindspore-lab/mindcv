"""
Generate a folder containing all the main variables' value.

P.S. This generated folder can be used by difference.py to calculate the difference statistics.

"""

import argparse
import os
import sys

from PIL import Image
from src.open_clip import create_model_and_transforms, get_tokenizer

import mindspore as ms
from mindspore import Tensor, ops


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=int,
        default=0,
        help="Mode of set_context, GRAPH_MODE(0) or PYNATIVE_MODE(1)",
    )
    parser.add_argument(
        "--device_target",
        type=str,
        default="Ascend",
        help="Ascend, CPU or GPU",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="A keyword (refer to ./src/open_clip/pretrained.py) or path of ckpt file",
    )
    parser.add_argument(
        "--quickgelu",
        type=bool,
        default=False,
        help="",
    )
    args = parser.parse_args(args)
    return args


def main(args):
    args = parse_args(args)
    ms.set_context(device_target=args.device_target, mode=args.mode)
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        args.model_name,
        args.pretrained,
        force_quick_gelu=args.quickgelu,
        force_custom_text=False,
        force_patch_dropout=None,
        force_image_size=None,
        image_mean=None,
        image_std=None,
        aug_cfg={},
    )
    tokenizer = get_tokenizer(args.model_name)

    image = Tensor(preprocess_val(Image.open("CLIP.png")))
    text = tokenizer(["a diagram", "a dog", "a cat"])

    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    if not os.path.exists("./" + args.model_name):
        root = "./" + args.model_name
        os.mkdir(root)

    file = open(root + "image.txt", "w+")
    file.write(str(image.asnumpy().tolist()))
    file.close()

    file = open(root + "text.txt", "w+")
    file.write(str(text.asnumpy().tolist()))
    file.close()

    file = open(root + "image_features.txt", "w+")
    file.write(str(image_features.asnumpy().tolist()))
    file.close()

    file = open(root + "text_features.txt", "w+")
    file.write(str(text_features.asnumpy().tolist()))
    file.close()

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = ops.softmax(100.0 * image_features @ text_features.T, axis=-1)

    file = open(root + "text_probs.txt", "w+")
    file.write(str(text_probs.asnumpy().tolist()))
    file.close()


if __name__ == "__main__":
    main(sys.argv[1:])
