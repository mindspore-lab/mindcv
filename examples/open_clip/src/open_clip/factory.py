import json
import logging
import os
import re
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from mindspore import load_checkpoint, load_param_into_net

from .constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from .loss import ClipLoss, DistillClipLoss
from .model import (
    CLIP,
    CustomTextCLIP,
    convert_to_custom_text_param_dict,
    convert_weights_to_lp,
    resize_pos_embed,
    resize_text_pos_embed,
)
from .openai import load_openai_model
from .pretrained import download_pretrained, get_pretrained_cfg, list_pretrained_tags_by_model
from .tokenizer import block_mask_tokenize, random_mask_tokenize, syntax_mask_tokenize, tokenize
from .transform import AugmentationCfg, image_transform

_MODEL_CONFIG_PATHS = [Path(__file__).parent / "model_configs/"]
_MODEL_CONFIGS = {}  # directory (model_name: config) of model architecture configs


def _natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", string_.lower())]


def _rescan_model_configs():
    global _MODEL_CONFIGS

    config_ext = (".json",)
    config_files = []
    for config_path in _MODEL_CONFIG_PATHS:
        if config_path.is_file() and config_path.suffix in config_ext:
            config_files.append(config_path)
        elif config_path.is_dir():
            for ext in config_ext:
                config_files.extend(config_path.glob(f"*{ext}"))

    for cf in config_files:
        with open(cf, "r") as f:
            model_cfg = json.load(f)
            if all(a in model_cfg for a in ("embed_dim", "vision_cfg", "text_cfg")):
                _MODEL_CONFIGS[cf.stem] = model_cfg

    _MODEL_CONFIGS = {k: v for k, v in sorted(_MODEL_CONFIGS.items(), key=lambda x: _natural_key(x[0]))}


_rescan_model_configs()  # initial populate of model config registry


def list_models():
    """enumerate available model architectures based on config files"""
    return list(_MODEL_CONFIGS.keys())


def add_model_config(path):
    """add model config path or file and update registry"""
    if not isinstance(path, Path):
        path = Path(path)
    _MODEL_CONFIG_PATHS.append(path)
    _rescan_model_configs()


def get_model_config(model_name):
    if model_name in _MODEL_CONFIGS:
        return deepcopy(_MODEL_CONFIGS[model_name])
    else:
        return None


def get_tokenizer(model_name):
    config = get_model_config(model_name)
    if "text_mask" in config["text_cfg"] and config["text_cfg"]["text_mask"] == "syntax":
        tokenizer = syntax_mask_tokenize
    elif "text_mask" in config["text_cfg"] and config["text_cfg"]["text_mask"] == "random":
        tokenizer = random_mask_tokenize
    elif "text_mask" in config["text_cfg"] and config["text_cfg"]["text_mask"] == "block":
        tokenizer = block_mask_tokenize
    else:
        tokenizer = tokenize
    if "context_length" in config["text_cfg"].keys():
        context_length = config["text_cfg"]["context_length"]
        tokenizer = partial(tokenizer, context_length=context_length)
    return tokenizer


def load_ckpt(model, checkpoint_path, strict=False):
    param_dict = load_checkpoint(checkpoint_path)
    # detect old format and make compatible with new format
    if "positional_embedding" in param_dict and not hasattr(model, "positional_embedding"):
        param_dict = convert_to_custom_text_param_dict(param_dict)
    position_id_key = "text.transformer.embeddings.position_ids"
    if position_id_key in param_dict and not hasattr(model, position_id_key):
        del param_dict[position_id_key]
    resize_pos_embed(param_dict, model)
    resize_text_pos_embed(param_dict, model)
    incompatible_keys = load_param_into_net(model, param_dict, strict_load=strict)
    return incompatible_keys


def create_model(
    model_name: str,
    pretrained: Optional[str] = None,
    force_quick_gelu: bool = False,
    force_custom_text: bool = False,
    force_patch_dropout: Optional[float] = None,
    force_image_size: Optional[Union[int, Tuple[int, int]]] = None,
    cache_dir: Optional[str] = None,
    require_pretrained: bool = False,
    **model_kwargs,
):
    model_name = model_name.replace("/", "-")  # for callers using old naming with / in ViT names
    pretrained_cfg = {}
    model_cfg = None

    if pretrained and pretrained.lower() == "openai":
        logging.info(f"Loading pretrained {model_name} from OpenAI.")
        model = load_openai_model(
            model_name,
            cache_dir=cache_dir,
        )
    else:
        model_cfg = model_cfg or get_model_config(model_name)
        if model_cfg is not None:
            logging.info(f"Loaded {model_name} model config.")
        else:
            logging.error(f"Model config for {model_name} not found; available models {list_models()}.")
            raise RuntimeError(f"Model config for {model_name} not found.")

        if force_quick_gelu:
            # override for use of QuickGELU on non-OpenAI transformer models
            model_cfg["quick_gelu"] = True

        if force_patch_dropout is not None:
            # override the default patch dropout value
            model_cfg["vision_cfg"]["patch_dropout"] = force_patch_dropout

        if force_image_size is not None:
            # override model config's image size
            model_cfg["vision_cfg"]["image_size"] = force_image_size

        custom_text = model_cfg.pop("custom_text", False) or force_custom_text
        if custom_text:
            if "coca" in model_name:
                raise ImportError("COCA model have not been supported yet.")
            else:
                model = CustomTextCLIP(**model_cfg, **model_kwargs)
        else:
            model = CLIP(**model_cfg, **model_kwargs)

        convert_weights_to_lp(model)

        pretrained_loaded = False
        if pretrained:
            checkpoint_path = ""
            pretrained_cfg = get_pretrained_cfg(model_name, pretrained)
            if pretrained_cfg:
                checkpoint_path = download_pretrained(pretrained_cfg, cache_dir=cache_dir)
            elif os.path.exists(pretrained):
                checkpoint_path = pretrained

            if checkpoint_path:
                logging.info(f"Loading pretrained {model_name} weights ({pretrained}).")
                load_ckpt(model, checkpoint_path)
            else:
                error_str = (
                    f"Pretrained weights ({pretrained}) not found for model {model_name}."
                    f"Available pretrained tags ({list_pretrained_tags_by_model(model_name)}."
                )
                logging.warning(error_str)
                raise RuntimeError(error_str)
            pretrained_loaded = True

        if require_pretrained and not pretrained_loaded:
            # callers of create_model_from_pretrained always expect pretrained weights
            raise RuntimeError(
                f"Pretrained weights were required for (model: {model_name}, pretrained: {pretrained}) but not loaded."
            )

        # set image / mean metadata from pretrained_cfg if available, or use default
        model.visual.image_mean = pretrained_cfg.get("mean", None) or OPENAI_DATASET_MEAN
        model.visual.image_std = pretrained_cfg.get("std", None) or OPENAI_DATASET_STD

    return model


def create_loss(args):
    if args.distill:
        return DistillClipLoss(
            local_loss=args.local_loss,
            gather_with_grad=args.gather_with_grad,
            cache_labels=True,
            rank=args.rank,
            world_size=args.world_size,
        )
    return ClipLoss(
        local_loss=args.local_loss,
        gather_with_grad=args.gather_with_grad,
        cache_labels=True,
        rank=args.rank,
        world_size=args.world_size,
    )


def create_model_and_transforms(
    model_name: str,
    pretrained: Optional[str] = None,
    force_quick_gelu: bool = False,
    force_custom_text: bool = False,
    force_patch_dropout: Optional[float] = None,
    force_image_size: Optional[Union[int, Tuple[int, int]]] = None,
    image_mean: Optional[Tuple[float, ...]] = None,
    image_std: Optional[Tuple[float, ...]] = None,
    aug_cfg: Optional[Union[Dict[str, Any], AugmentationCfg]] = None,
    cache_dir: Optional[str] = None,
    **model_kwargs,
):
    model = create_model(
        model_name,
        pretrained,
        force_quick_gelu=force_quick_gelu,
        force_custom_text=force_custom_text,
        force_patch_dropout=force_patch_dropout,
        force_image_size=force_image_size,
        cache_dir=cache_dir,
        **model_kwargs,
    )

    image_mean = image_mean or getattr(model.visual, "image_mean", None)
    image_std = image_std or getattr(model.visual, "image_std", None)
    preprocess_train = image_transform(
        model.visual.image_size,
        is_train=True,
        mean=image_mean,
        std=image_std,
        aug_cfg=aug_cfg,
    )
    preprocess_val = image_transform(
        model.visual.image_size,
        is_train=False,
        mean=image_mean,
        std=image_std,
    )

    return model, preprocess_train, preprocess_val
