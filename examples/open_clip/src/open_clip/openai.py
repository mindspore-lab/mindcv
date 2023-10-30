""" OpenAI pretrained model functions

Adapted from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""

import os
from typing import List, Optional

import mindspore as ms

from .constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from .model import build_model_from_openai_ckpt
from .pretrained import download_pretrained_from_url, get_pretrained_url, list_pretrained_models_by_tag

__all__ = ["list_openai_models", "load_openai_model"]


def list_openai_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return list_pretrained_models_by_tag("openai")


def load_openai_model(
    name: str,
    cache_dir: Optional[str] = None,
):
    """Load a CLIP model

    Parameters
    ----------
    name : str
        A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the param_dict
    cache_dir : Optional[str]
        The directory to cache the downloaded model weights

    Returns
    -------
    model : mindspore.nn.Cell
        The CLIP model
    preprocess : Callable[[PIL.Image], mindspore.Tensor]
        Transform operations that converts a PIL image into a tensor that the returned model can take as its input
    """

    if get_pretrained_url(name, "openai"):
        model_path = download_pretrained_from_url(get_pretrained_url(name, "openai"), cache_dir=cache_dir)
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {list_openai_models()}")

    param_dict = ms.load_checkpoint(model_path)

    try:
        model = build_model_from_openai_ckpt(param_dict)
    except KeyError:
        sd = {k[7:]: v for k, v in param_dict["state_dict"].items()}
        model = build_model_from_openai_ckpt(sd)

    # add mean / std attributes for consistency with OpenCLIP models
    model.visual.image_mean = OPENAI_DATASET_MEAN
    model.visual.image_std = OPENAI_DATASET_STD
    return model
