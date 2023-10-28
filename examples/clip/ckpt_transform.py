import argparse
import os
import sys

import torch

import mindspore as ms

from examples.clip.clip.clip import _MODELS, _download


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pth_path",
        type=str,
        default=None,
        help="Model name or the path of the model's checkpoint file given by OpenAI",
    )
    args = parser.parse_args(args)
    return args


def pytorch_params(pth_file):
    par_dict = torch.load(pth_file, map_location="cpu").state_dict()
    pt_params = []
    for name in par_dict:
        parameter = par_dict[name]
        if "ln_" in name:
            name = name.replace(".weight", ".gamma").replace(".bias", ".beta")
        elif name == "token_embedding.weight":
            name = "token_embedding.embedding_table"
        elif ".bn" in name or ".downsample.1." in name:
            name = name.replace(".weight", ".gamma").replace(".bias", ".beta")
            name = name.replace(".running_mean", ".moving_mean").replace(".running_var", ".moving_variance")
        pt_params.append({"name": name, "data": ms.Tensor(parameter.numpy())})
    return pt_params


def main(args):
    args = parse_args(args)
    if os.path.exists(args.pth_path):
        pt_param = pytorch_params(args.pth_path)
        ms.save_checkpoint(pt_param, args.pth_path.replace(".pt", ".ckpt"))
    elif args.pth_path in _MODELS.keys():
        model_path = _download(_MODELS[args.pth_path], os.path.expanduser("~/"))
        pt_param = pytorch_params(model_path)
        ms.save_checkpoint(pt_param, os.path.expanduser("~/"))
    else:
        raise ValueError(
            f"{args.pth_path} is not a supported checkpoint file or model name. "
            f"Models with available checkpoint file are: {list(_MODELS.keys())}"
        )
    print("Done!")


if __name__ == "__main__":
    main(sys.argv[1:])
