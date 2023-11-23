""" CLIP Model

Adapted from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""
import logging
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np

import mindspore as ms
from mindspore import Parameter, Tensor, load_param_into_net, nn, ops

from .modified_resnet import ModifiedResNet
from .transformer import Attention, QuickGELU, TextTransformer, VisionTransformer
from .utils import to_2tuple


@dataclass
class CLIPVisionCfg:
    layers: Union[Tuple[int, int, int, int], int] = 12
    width: int = 768
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 16
    image_size: Union[Tuple[int, int], int] = 224

    ls_init_value: Optional[float] = None  # layer scale initial value
    patch_dropout: float = 0.0  # what fraction of patches to dropout during training (0 would mean disabled and no patches dropped) - 0.5 to 0.75 recommended in the paper for optimal results # noqa
    input_patchnorm: bool = False  # whether to use dual patchnorm - would only apply the input layernorm on each patch, as post-layernorm already exist in original clip vit design # noqa
    global_average_pool: bool = False  # whether to global average pool the last embedding layer, instead of using CLS token (https://arxiv.org/abs/2205.01580) # noqa
    attentional_pool: bool = False  # whether to use attentional pooler in the last embedding layer
    n_queries: int = 256  # n_queries for attentional pooler
    attn_pooler_heads: int = 8  # n heads for attentional_pooling
    output_tokens: bool = False


@dataclass
class CLIPTextCfg:
    context_length: int = 77
    vocab_size: int = 49408
    width: int = 512
    heads: int = 8
    layers: int = 12
    ls_init_value: Optional[float] = None  # layer scale initial value
    proj: str = "mlp"
    pooler_type: str = "mean_pooler"
    embed_cls: bool = False
    pad_id: int = 0
    output_tokens: bool = False
    text_mask: str = "first"  # default first truncate in bpe_tokenizer


def get_input_dtype(precision: str):
    input_dtype = None
    if precision == "fp32":
        input_dtype = ms.float32
    elif precision == "fp16":
        input_dtype = ms.float16
    return input_dtype


def _build_vision_tower(embed_dim: int, vision_cfg: CLIPVisionCfg, quick_gelu: bool = False):
    if isinstance(vision_cfg, dict):
        vision_cfg = CLIPVisionCfg(**vision_cfg)

    act_layer = QuickGELU if quick_gelu else nn.GELU(approximate=False)

    if isinstance(vision_cfg.layers, (tuple, list)):
        vision_heads = vision_cfg.width * 32 // vision_cfg.head_width
        visual = ModifiedResNet(
            layers=vision_cfg.layers,
            output_dim=embed_dim,
            heads=vision_heads,
            image_size=vision_cfg.image_size,
            width=vision_cfg.width,
        )
    else:
        vision_heads = vision_cfg.width // vision_cfg.head_width
        visual = VisionTransformer(
            image_size=vision_cfg.image_size,
            patch_size=vision_cfg.patch_size,
            width=vision_cfg.width,
            layers=vision_cfg.layers,
            heads=vision_heads,
            mlp_ratio=vision_cfg.mlp_ratio,
            ls_init_value=vision_cfg.ls_init_value,
            patch_dropout=vision_cfg.patch_dropout,
            input_patchnorm=vision_cfg.input_patchnorm,
            global_average_pool=vision_cfg.global_average_pool,
            attentional_pool=vision_cfg.attentional_pool,
            n_queries=vision_cfg.n_queries,
            attn_pooler_heads=vision_cfg.attn_pooler_heads,
            output_tokens=vision_cfg.output_tokens,
            output_dim=embed_dim,
            act_layer=act_layer,
        )

    return visual


def _build_text_tower(
    embed_dim: int,
    text_cfg: CLIPTextCfg,
    quick_gelu: bool = False,
):
    if isinstance(text_cfg, dict):
        text_cfg = CLIPTextCfg(**text_cfg)

    act_layer = QuickGELU if quick_gelu else nn.GELU(approximate=False)

    text = TextTransformer(
        context_length=text_cfg.context_length,
        vocab_size=text_cfg.vocab_size,
        width=text_cfg.width,
        heads=text_cfg.heads,
        layers=text_cfg.layers,
        ls_init_value=text_cfg.ls_init_value,
        output_dim=embed_dim,
        embed_cls=text_cfg.embed_cls,
        output_tokens=text_cfg.output_tokens,
        pad_id=text_cfg.pad_id,
        act_layer=act_layer,
    )
    return text


class CLIP(nn.Cell):
    def __init__(
        self,
        embed_dim: int,
        vision_cfg: CLIPVisionCfg,
        text_cfg: CLIPTextCfg,
        quick_gelu: bool = False,
        init_logit_scale: float = np.log(1 / 0.07),
        init_logit_bias: Optional[float] = None,
    ):
        super().__init__()
        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu)
        text = _build_text_tower(embed_dim, text_cfg, quick_gelu)
        self.transformer = text.transformer
        self.context_length = text.context_length
        self.vocab_size = text.vocab_size
        self.token_embedding = text.token_embedding
        self.positional_embedding = text.positional_embedding
        self.ln_final = text.ln_final
        self.text_projection = text.text_projection
        self.attn_mask = text.attn_mask
        self.logit_scale = Parameter(ops.ones([]) * init_logit_scale)
        if init_logit_bias is not None:
            self.logit_bias = Parameter(ops.ones([]) * init_logit_bias)
        else:
            self.logit_bias = None

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.transformer.grad_checkpointing = enable

    def encode_image(self, image, normalize: bool = False):
        features = self.visual(image)
        return ops.L2Normalize(-1)(features) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)
        x = x.permute((1, 0, 2))  # NLD -> LND
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute((1, 0, 2))  # LND -> NLD
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[ops.arange(x.shape[0]), text.argmax(axis=-1)] @ self.text_projection.to(x.dtype)
        return ops.L2Normalize(-1)(x) if normalize else x

    def construct(self, image: Optional[Tensor] = None, text: Optional[Tensor] = None):
        image_features = self.encode_image(image, normalize=True) if image is not None else None
        text_features = self.encode_text(text, normalize=True) if text is not None else None

        if self.logit_bias is not None:
            return image_features, text_features, self.logit_scale.exp(), self.logit_bias
        return image_features, text_features, self.logit_scale.exp()


class CustomTextCLIP(nn.Cell):
    def __init__(
        self,
        embed_dim: int,
        vision_cfg: CLIPVisionCfg,
        text_cfg: CLIPTextCfg,
        init_logit_scale: float = np.log(1 / 0.07),
        init_logit_bias: Optional[float] = None,
        quick_gelu: bool = False,
    ):
        super().__init__()
        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu)
        self.text = _build_text_tower(embed_dim, text_cfg, quick_gelu)
        self.context_length = self.text.context_length
        self.vocab_size = self.text.vocab_size
        self.logit_scale = Parameter(ops.ones([]) * init_logit_scale)
        if init_logit_bias is not None:
            self.logit_bias = Parameter(ops.ones([]) * init_logit_bias)
        else:
            self.logit_bias = None

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    def lock_text_tower(self, unlocked_layers: int = 0, freeze_layer_norm: bool = True):
        self.text.lock(unlocked_layers, freeze_layer_norm)

    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.text.set_grad_checkpointing(enable)

    def encode_image(self, image, normalize: bool = False):
        features = self.visual(image)
        return ops.L2Normalize(-1)(features) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        features = self.text(text)
        return ops.L2Normalize(-1)(features) if normalize else features

    def construct(
        self,
        image: Optional[Tensor] = None,
        text: Optional[Tensor] = None,
    ):
        image_features = self.encode_image(image, normalize=True) if image is not None else None
        text_features = self.encode_text(text, normalize=True) if text is not None else None

        if self.logit_bias is not None:
            return image_features, text_features, self.logit_scale.exp(), self.logit_bias
        return image_features, text_features, self.logit_scale.exp()


def convert_weights_to_lp(model: nn.Cell):
    """Convert applicable model parameters to fp16"""

    def _convert_weights(cell):
        if isinstance(cell, (nn.Conv1d, nn.Conv2d, nn.Dense, Attention)):
            cell.to_float(ms.float16)

    model.apply(_convert_weights)


convert_weights_to_fp16 = convert_weights_to_lp  # backwards compat


# used to maintain checkpoint compatibility
def convert_to_custom_text_param_dict(param_dict: dict):
    if "text_projection" in param_dict:
        # old format param_dict, move text tower -> .text
        # this might happen when use ckpt from OpenAI-CLIP
        new_param_dict = {}
        for k, v in param_dict.items():
            if any(
                k.startswith(p)
                for p in (
                    "text_projection",
                    "positional_embedding",
                    "token_embedding",
                    "transformer",
                    "ln_final",
                )
            ):
                k = "text." + k
            new_param_dict[k] = v
        return new_param_dict
    return param_dict


def build_model_from_openai_ckpt(
    param_dict: dict,
    quick_gelu=True,
):
    vit = "visual.proj" in param_dict

    if vit:
        vision_width = param_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in param_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")]
        )
        vision_patch_size = param_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((param_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_size = vision_patch_size * grid_size
    else:
        counts: list = [
            len(set(k.split(".")[2] for k in param_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]
        ]
        vision_layers = tuple(counts)
        vision_width = param_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((param_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width**2 + 1 == param_dict["visual.attnpool.positional_embedding"].shape[0]
        image_size = output_width * 32

    embed_dim = param_dict["text_projection"].shape[1]
    context_length = param_dict["positional_embedding"].shape[0]
    vocab_size = param_dict["token_embedding.embedding_table"].shape[0]
    transformer_width = param_dict["ln_final.gamma"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in param_dict if k.startswith("transformer.resblocks")))

    vision_cfg = CLIPVisionCfg(
        layers=vision_layers,
        width=vision_width,
        patch_size=vision_patch_size,
        image_size=image_size,
    )
    text_cfg = CLIPTextCfg(
        context_length=context_length,
        vocab_size=vocab_size,
        width=transformer_width,
        heads=transformer_heads,
        layers=transformer_layers,
    )
    model = CLIP(embed_dim, vision_cfg=vision_cfg, text_cfg=text_cfg, quick_gelu=quick_gelu)

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in param_dict:
            del param_dict[key]

    convert_weights_to_fp16(model)  # OpenAI state dicts are partially converted to float16
    load_param_into_net(model, param_dict)
    return model.set_train(False)


def resize_pos_embed(param_dict, model, interpolation: str = "bicubic"):
    # Rescale the grid of position embeddings when loading from param_dict
    old_pos_embed = param_dict.get("visual.positional_embedding", None)
    if old_pos_embed is None or not hasattr(model.visual, "grid_size"):
        return
    grid_size = to_2tuple(model.visual.grid_size)
    extra_tokens = 1  # TODO: detect different token configs (ie no class token, or more tokens)
    new_seq_len = grid_size[0] * grid_size[1] + extra_tokens
    if new_seq_len == old_pos_embed.shape[0]:
        return

    if extra_tokens:
        pos_emb_tok, pos_emb_img = old_pos_embed[:extra_tokens], old_pos_embed[extra_tokens:]
    else:
        pos_emb_tok, pos_emb_img = None, old_pos_embed
    old_grid_size = to_2tuple(int(math.sqrt(len(pos_emb_img))))

    logging.info("Resizing position embedding grid-size from %s to %s", old_grid_size, grid_size)
    pos_emb_img = pos_emb_img.reshape(1, old_grid_size[0], old_grid_size[1], -1).permute(0, 3, 1, 2)
    pos_emb_img = ops.interpolate(
        pos_emb_img,
        size=grid_size,
        mode=interpolation,
        align_corners=False,
    )

    pos_emb_img = pos_emb_img.permute(0, 2, 3, 1).reshape(1, grid_size[0] * grid_size[1], -1)[0]
    if pos_emb_tok is not None:
        new_pos_embed = ops.cat([pos_emb_tok.to(pos_emb_img.dtype), pos_emb_img], axis=0)
    else:
        new_pos_embed = pos_emb_img
    param_dict["visual.positional_embedding"] = new_pos_embed


def resize_text_pos_embed(param_dict, model, interpolation: str = "linear"):
    old_pos_embed = param_dict.get("positional_embedding", None)
    if old_pos_embed is None:
        return
    # FIXME add support for text cls_token
    model_pos_embed = getattr(model, "positional_embedding", None)
    if model_pos_embed is None:
        model_pos_embed = getattr(model.text, "positional_embedding", None)

    old_num_pos = old_pos_embed.shape[0]
    old_width = old_pos_embed.shape[1]
    num_pos = model_pos_embed.shape[0]
    width = model_pos_embed.shape[1]
    assert old_width == width, "text pos_embed width changed!"
    if old_num_pos == num_pos:
        return

    logging.info("Resizing text position embedding num_pos from %s to %s", old_num_pos, num_pos)
    old_pos_embed = old_pos_embed.reshape(1, old_num_pos, old_width).permute(0, 2, 1)
    old_pos_embed = ops.interpolate(
        old_pos_embed,
        size=num_pos,
        mode=interpolation,
        align_corners=False,
    )
    old_pos_embed = old_pos_embed.permute(0, 2, 1)[0]
    new_pos_embed = old_pos_embed

    param_dict["positional_embedding"] = new_pos_embed
