"""ViT"""
import functools
from typing import Callable, Optional

import numpy as np

import mindspore as ms
from mindspore import Parameter, Tensor, nn, ops
from mindspore.common.initializer import TruncatedNormal, XavierUniform, initializer

from .helpers import load_pretrained
from .layers.compatibility import Dropout
from .layers.drop_path import DropPath
from .layers.mlp import Mlp
from .layers.patch_dropout import PatchDropout
from .layers.patch_embed import PatchEmbed
from .layers.pos_embed import resample_abs_pos_embed
from .registry import register_model

__all__ = [
    "VisionTransformer",
    "vit_b_16_224",
    "vit_b_16_384",
    "vit_l_16_224",  # with pretrained weights
    "vit_l_16_384",
    "vit_b_32_224",  # with pretrained weights
    "vit_b_32_384",
    "vit_l_32_224",  # with pretrained weights
]


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 224, 224),
        "first_conv": "patch_embed.proj",
        "classifier": "head",
        **kwargs,
    }


default_cfgs = {
    "vit_b_16_224": _cfg(url=""),
    "vit_b_16_384": _cfg(
        url="", input_size=(3, 384, 384)
    ),
    "vit_l_16_224": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/vit/vit_l_16_224-d2635f8b.ckpt"),
    "vit_l_16_384": _cfg(
        url="", input_size=(3, 384, 384)
    ),
    "vit_b_32_224": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/vit/vit_b_32_224-4a1c9d8e.ckpt"),
    "vit_b_32_384": _cfg(
        url="", input_size=(3, 384, 384)
    ),
    "vit_l_32_224": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/vit/vit_l_32_224-8c8ea164.ckpt"),
}


# TODO: Flash Attention
class Attention(nn.Cell):
    """
    Attention layer implementation, Rearrange Input -> B x N x hidden size.

    Args:
        dim (int): The dimension of input features.
        num_heads (int): The number of attention heads. Default: 8.
        qkv_bias (bool): Specifies whether the linear layer uses a bias vector. Default: True.
        qk_norm (bool): Specifies whether to do normalization to q and k.
        attn_drop (float): The drop rate of attention, greater than 0 and less equal than 1. Default: 0.0.
        proj_drop (float): The drop rate of output, greater than 0 and less equal than 1. Default: 0.0.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> ops = Attention(768, 12)
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Cell = nn.LayerNorm,
    ):
        super(Attention, self).__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = Tensor(self.head_dim ** -0.5)

        self.qkv = nn.Dense(dim, dim * 3, has_bias=qkv_bias)
        self.q_norm = norm_layer((self.head_dim,)) if qk_norm else nn.Identity()
        self.k_norm = norm_layer((self.head_dim,)) if qk_norm else nn.Identity()

        self.attn_drop = Dropout(attn_drop)
        self.proj = nn.Dense(dim, dim)
        self.proj_drop = Dropout(proj_drop)

        self.mul = ops.Mul()
        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()
        self.unstack = ops.Unstack(axis=0)
        self.attn_matmul_v = ops.BatchMatMul()
        self.q_matmul_k = ops.BatchMatMul(transpose_b=True)

    def construct(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x)
        qkv = self.reshape(qkv, (b, n, 3, self.num_heads, self.head_dim))
        qkv = self.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = self.unstack(qkv)
        q, k = self.q_norm(q), self.k_norm(k)

        q = self.mul(q, self.scale**0.5)
        k = self.mul(k, self.scale**0.5)
        attn = self.q_matmul_k(q, k)

        attn = ops.softmax(attn.astype(ms.float32), axis=-1).astype(attn.dtype)
        attn = self.attn_drop(attn)

        out = self.attn_matmul_v(attn, v)
        out = self.transpose(out, (0, 2, 1, 3))
        out = self.reshape(out, (b, n, c))
        out = self.proj(out)
        out = self.proj_drop(out)

        return out


class LayerScale(nn.Cell):
    """
    Layer scale, help ViT improve the training dynamic, allowing for the training
    of deeper high-capacity image transformers that benefit from depth

    Args:
        dim (int): The output dimension of attnetion layer or mlp layer.
        init_values (float): The scale factor. Default: 1e-5.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> ops = LayerScale(768, 0.01)
    """
    def __init__(
        self,
        dim: int,
        init_values: float = 1e-5
    ):
        super(LayerScale, self).__init__()
        self.gamma = Parameter(initializer(init_values, dim))

    def construct(self, x):
        return self.gamma * x


class Block(nn.Cell):
    """
    Transformer block implementation.

    Args:
        dim (int): The dimension of embedding.
        num_heads (int): The number of attention heads.
        qkv_bias (bool): Specifies whether the linear layer uses a bias vector. Default: True.
        attn_drop (float): The drop rate of attention, greater than 0 and less equal than 1. Default: 0.0.
        proj_drop (float): The drop rate of dense layer output, greater than 0 and less equal than 1. Default: 0.0.
        mlp_ratio (float): The ratio used to scale the input dimensions to obtain the dimensions of the hidden layer.
        drop_path (float): The drop rate for drop path. Default: 0.0.
        act_layer (nn.Cell): Activation function which will be stacked on top of the
            normalization layer (if not None), otherwise on top of the conv layer. Default: nn.GELU.
        norm_layer (nn.Cell): Norm layer that will be stacked on top of the convolution
            layer. Default: nn.LayerNorm.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> ops = TransformerEncoder(768, 12, 12, 3072)
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_drop: float = 0.,
        attn_drop: float = 0.,
        init_values: Optional[float] = None,
        drop_path: float = 0.,
        act_layer: nn.Cell = nn.GELU,
        norm_layer: nn.Cell = nn.LayerNorm,
        mlp_layer: Callable = Mlp,
    ):
        super(Block, self).__init__()
        self.norm1 = norm_layer((dim,))
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim=dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer((dim,))
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop
        )
        self.ls2 = LayerScale(dim=dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def construct(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class VisionTransformer(nn.Cell):
    '''
    ViT encoder, which returns the feature encoded by transformer encoder.
    '''
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        global_pool: str = 'token',
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        drop_rate: float = 0.,
        pos_drop_rate: float = 0.,
        patch_drop_rate: float = 0.,
        proj_drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.,
        weight_init: bool = True,
        init_values: Optional[float] = None,
        no_embed_class: bool = False,
        pre_norm: bool = False,
        fc_norm: Optional[bool] = None,
        dynamic_img_size: bool = False,
        dynamic_img_pad: bool = False,
        act_layer: nn.Cell = nn.GELU,
        embed_layer: Callable = PatchEmbed,
        norm_layer: nn.Cell = nn.LayerNorm,
        mlp_layer: Callable = Mlp,
        class_token: bool = True,
        block_fn: Callable = Block,
        num_classes: int = 1000,
    ):
        super(VisionTransformer, self).__init__()
        assert global_pool in ('', 'avg', 'token')
        assert class_token or global_pool != 'token'
        use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm

        self.global_pool = global_pool
        self.num_prefix_tokens = 1 if class_token else 0
        self.no_embed_class = no_embed_class
        self.dynamic_img_size = dynamic_img_size
        self.dynamic_img_pad = dynamic_img_pad

        embed_args = {}
        if dynamic_img_size:
            # flatten deferred until after pos embed
            embed_args.update(dict(strict_img_size=False, output_fmt='NHWC'))
        elif dynamic_img_pad:
            embed_args.update(dict(output_fmt='NHWC'))

        self.patch_embed = embed_layer(
            image_size=image_size,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=embed_dim,
            bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
            dynamic_img_pad=dynamic_img_pad,
            **embed_args,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = Parameter(initializer(TruncatedNormal(0.02), (1, 1, embed_dim))) if class_token else None
        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        self.pos_embed = Parameter(initializer(TruncatedNormal(0.02), (1, embed_len, embed_dim)))
        self.pos_drop = Dropout(pos_drop_rate)
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
            )
        else:
            self.patch_drop = nn.Identity()

        self.norm_pre = norm_layer((embed_dim,)) if pre_norm else nn.Identity()
        dpr = [x.item() for x in np.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.CellList([
            block_fn(
                dim=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm,
                attn_drop=attn_drop_rate, proj_drop=proj_drop_rate,
                mlp_ratio=mlp_ratio, drop_path=dpr[i], init_values=init_values,
                act_layer=act_layer, norm_layer=norm_layer, mlp_layer=mlp_layer,
            ) for i in range(depth)
        ])

        self.norm = norm_layer((embed_dim,)) if not use_fc_norm else nn.Identity()
        self.fc_norm = norm_layer((embed_dim,)) if use_fc_norm else nn.Identity()
        self.head_drop = Dropout(drop_rate)
        self.head = nn.Dense(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if weight_init:
            self._init_weights()

    def get_num_layers(self):
        return len(self.blocks)

    def _init_weights(self):
        w = self.patch_embed.proj.weight
        w_shape_flatted = (w.shape[0], functools.reduce(lambda x, y: x*y, w.shape[1:]))
        w.set_data(initializer(XavierUniform(), w_shape_flatted, w.dtype).reshape(w.shape))
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(
                    initializer(XavierUniform(), cell.weight.shape, cell.weight.dtype)
                )
                if cell.bias is not None:
                    cell.bias.set_data(
                        initializer('zeros', cell.bias.shape, cell.bias.dtype)
                    )
            elif isinstance(cell, nn.LayerNorm):
                cell.gamma.set_data(
                    initializer('ones', cell.gamma.shape, cell.gamma.dtype)
                )
                cell.beta.set_data(
                    initializer('zeros', cell.beta.shape, cell.beta.dtype)
                )

    def _pos_embed(self, x):
        if self.dynamic_img_size or self.dynamic_img_pad:
            # bhwc format
            B, H, W, C = x.shape
            pos_embed = resample_abs_pos_embed(
                self.pos_embed,
                (H, W),
                num_prefix_tokens=0 if self.no_embed_class else self.num_prefix_tokens,
            )
            x = ops.reshape(x, (B, -1, C))
        else:
            pos_embed = self.pos_embed

        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + pos_embed
            if self.cls_token is not None:
                cls_tokens = ops.broadcast_to(self.cls_token, (x.shape[0], -1, -1))
                cls_tokens = cls_tokens.astype(x.dtype)
                x = ops.concat((cls_tokens, x), axis=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if self.cls_token is not None:
                cls_tokens = ops.broadcast_to(self.cls_token, (x.shape[0], -1, -1))
                cls_tokens = cls_tokens.astype(x.dtype)
                x = ops.concat((cls_tokens, x), axis=1)
            x = x + pos_embed

        return self.pos_drop(x)

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def forward_head(self, x):
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(axis=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        x = self.head_drop(x)
        x = self.head(x)
        return x

    def construct(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


@register_model
def vit_b_16_224(pretrained: bool = False, num_classes: int = 1000, in_channels: int = 3, **kwargs):
    default_cfg = default_cfgs["vit_b_16_224"]
    model = VisionTransformer(
        image_size=224, patch_size=16, in_channels=in_channels, embed_dim=768, depth=12, num_heads=12,
        num_classes=num_classes, **kwargs
    )
    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def vit_b_16_384(pretrained: bool = False, num_classes: int = 1000, in_channels: int = 3, **kwargs):
    default_cfg = default_cfgs["vit_b_16_384"]
    model = VisionTransformer(
        image_size=384, patch_size=16, in_channels=in_channels, embed_dim=768, depth=12, num_heads=12,
        num_classes=num_classes, **kwargs
    )
    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def vit_l_16_224(pretrained: bool = False, num_classes: int = 1000, in_channels: int = 3, **kwargs):
    default_cfg = default_cfgs["vit_l_16_224"]
    model = VisionTransformer(
        image_size=224, patch_size=16, in_channels=in_channels, embed_dim=1024, depth=24, num_heads=16,
        num_classes=num_classes, **kwargs
    )
    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def vit_l_16_384(pretrained: bool = False, num_classes: int = 1000, in_channels: int = 3, **kwargs):
    default_cfg = default_cfgs["vit_l_16_384"]
    model = VisionTransformer(
        image_size=384, patch_size=16, in_channels=in_channels, embed_dim=1024, depth=24, num_heads=16,
        num_classes=num_classes, **kwargs
    )
    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def vit_b_32_224(pretrained: bool = False, num_classes: int = 1000, in_channels: int = 3, **kwargs):
    default_cfg = default_cfgs["vit_b_32_224"]
    model = VisionTransformer(
        image_size=224, patch_size=32, in_channels=in_channels, embed_dim=768, depth=12, num_heads=12,
        num_classes=num_classes, **kwargs
    )
    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def vit_b_32_384(pretrained: bool = False, num_classes: int = 1000, in_channels: int = 3, **kwargs):
    default_cfg = default_cfgs["vit_b_32_384"]
    model = VisionTransformer(
        image_size=384, patch_size=32, in_channels=in_channels, embed_dim=768, depth=12, num_heads=12,
        num_classes=num_classes, **kwargs
    )
    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def vit_l_32_224(pretrained: bool = False, num_classes: int = 1000, in_channels: int = 3, **kwargs):
    default_cfg = default_cfgs["vit_l_32_224"]
    model = VisionTransformer(
        image_size=224, patch_size=32, in_channels=in_channels, embed_dim=1024, depth=24, num_heads=16,
        num_classes=num_classes, **kwargs
    )
    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model
