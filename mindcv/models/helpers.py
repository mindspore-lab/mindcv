"""
Some utils while building models
"""
import collections.abc
import logging
import os
from itertools import repeat
from typing import Callable, Dict, List, Optional

import mindspore.nn as nn
from mindspore import load_checkpoint, load_param_into_net

from mindcv.utils.download import DownLoad, get_default_download_root

from .features import FeatureExtractWrapper


def get_checkpoint_download_root():
    return os.path.join(get_default_download_root(), "models")


class ConfigDict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def load_pretrained(model, default_cfg, num_classes=1000, in_channels=3, filter_fn=None):
    """load pretrained model depending on cfgs of model"""
    if "url" not in default_cfg or not default_cfg["url"]:
        logging.warning("Pretrained model URL is invalid")
        return

    # download files
    download_path = get_checkpoint_download_root()
    os.makedirs(download_path, exist_ok=True)
    DownLoad().download_url(default_cfg["url"], path=download_path)

    param_dict = load_checkpoint(os.path.join(download_path, os.path.basename(default_cfg["url"])))

    if in_channels == 1:
        conv1_name = default_cfg["first_conv"]
        logging.info("Converting first conv (%s) from 3 to 1 channel", conv1_name)
        con1_weight = param_dict[conv1_name + ".weight"]
        con1_weight.set_data(con1_weight.sum(axis=1, keepdims=True), slice_shape=True)
    elif in_channels != 3:
        raise ValueError("Invalid in_channels for pretrained weights")

    classifier_name = default_cfg["classifier"]
    if num_classes == 1000 and default_cfg["num_classes"] == 1001:
        classifier_weight = param_dict[classifier_name + ".weight"]
        classifier_weight.set_data(classifier_weight[:1000], slice_shape=True)
        classifier_bias = param_dict[classifier_name + ".bias"]
        classifier_bias.set_data(classifier_bias[:1000], slice_shape=True)
    elif num_classes != default_cfg["num_classes"]:
        params_names = list(param_dict.keys())
        param_dict.pop(
            _search_param_name(params_names, classifier_name + ".weight"),
            "No Parameter {} in ParamDict".format(classifier_name + ".weight"),
        )
        param_dict.pop(
            _search_param_name(params_names, classifier_name + ".bias"),
            "No Parameter {} in ParamDict".format(classifier_name + ".bias"),
        )

    if filter_fn is not None:
        param_dict = filter_fn(param_dict)

    load_param_into_net(model, param_dict)


def make_divisible(
    v: float,
    divisor: int,
    min_value: Optional[int] = None,
) -> int:
    """Find the smallest integer larger than v and divisible by divisor."""
    if not min_value:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


def _search_param_name(params_names: List, param_name: str) -> str:
    for pi in params_names:
        if param_name in pi:
            return pi
    return ""


def load_model_checkpoint(model: nn.Cell, checkpoint_path: str = "", ema: bool = False):
    """Model loads checkpoint.

    Args:
        model (nn.Cell): The model which loads the checkpoint.
        checkpoint_path (str): The path of checkpoint files. Default: "".
        ema (bool): Whether use ema method. Default: False.
    """

    if os.path.exists(checkpoint_path):
        checkpoint_param = load_checkpoint(checkpoint_path)
        ema_param_dict = dict()

        for param in checkpoint_param:
            if param.startswith("ema"):
                new_name = param.split("ema.")[1]
                ema_data = checkpoint_param[param]
                ema_data.name = new_name
                ema_param_dict[new_name] = ema_data

        if ema_param_dict and ema:
            load_param_into_net(model, ema_param_dict)
        elif bool(ema_param_dict) is False and ema:
            raise ValueError("chekpoint_param does not contain ema_parameter, please set ema is False.")
        else:
            load_param_into_net(model, checkpoint_param)


def build_model_with_cfg(
        model_cls: Callable,
        pretrained: bool,
        default_cfg: Dict,
        features_only: bool = False,
        out_indices: List[int] = [0, 1, 2, 3, 4],
        **kwargs,
):
    """Build model with specific model configurations

    Args:
        model_cls (nn.Cell): model class
        pretrained (bool): whether to load pretrained weights
        features_only (bool): Output the features at different strides instead. Default: False
        out_indices (list[int]): The indicies of the output features when `features_only` is `True`.
            Default: [0, 1, 2, 3, 4]
    """
    model = model_cls(**kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, kwargs.get("num_classes"), kwargs.get("in_channels"))

    if features_only:
        # wrap the model, output the feature pyramid instead
        try:
            model = FeatureExtractWrapper(model, out_indices=out_indices)
        except AttributeError as e:
            raise RuntimeError(f"`feature_only` is not implemented for `{model_cls.__name__}` model.") from e

    return model
