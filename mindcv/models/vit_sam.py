from collections import OrderedDict
from functools import partial
from typing import Callable, Optional, Tuple

import numpy as np

import mindspore as ms
from mindspore import Parameter, nn, ops
from mindspore.common.initializer import initializer

from .helpers import load_pretrained
from .layers.compatibility import Dropout
from .layers.drop_path import DropPath
from .layers.format import Format
from .layers.mlp import Mlp
from .layers.patch_dropout import PatchDropout
from .layers.patch_embed import PatchEmbed
from .layers.pooling import GlobalAvgPooling
from .registry import register_model

__all__ = [
    "VitSAM",
    "samvit_base_patch16",
    "samvit_large_patch16",
    "samvit_huge_patch16",
]


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 1024, 1024),
        "first_conv": "patch_embed.proj",
        "classifier": "head.classifier",
        **kwargs,
    }


default_cfgs = {
    "samvit_base_patch16": _cfg(
        url="", num_classes=0, input_size=(3, 1024, 1024), crop_pct=1.0
    ),
    "samvit_large_patch16": _cfg(
        url="", num_classes=0, input_size=(3, 1024, 1024), crop_pct=1.0
    ),
    "samvit_huge_patch16": _cfg(
        url="", num_classes=0, input_size=(3, 1024, 1024), crop_pct=1.0
    ),
}


class Attention(nn.Cell):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        qk_norm=False,
        attn_drop=0.,
        norm_layer=nn.LayerNorm,
        use_rel_pos: bool = False,
        input_size: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Dense(dim, dim * 3, has_bias=qkv_bias)
        self.q_norm = norm_layer((self.head_dim,)) if qk_norm else nn.Identity()
        self.k_norm = norm_layer((self.head_dim,)) if qk_norm else nn.Identity()
        self.attn_drop = Dropout(attn_drop)
        self.proj = nn.Dense(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                    input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            self.rel_pos_h = ms.Parameter(ops.zeros((2 * input_size[0] - 1, self.head_dim)))
            self.rel_pos_w = ms.Parameter(ops.zeros((2 * input_size[1] - 1, self.head_dim)))

        self.attn_matmul_v = ops.BatchMatMul()
        self.q_matmul_k = ops.BatchMatMul(transpose_b=True)

    def construct(self, x):
        b, h, w, _ = x.shape
        x = ops.reshape(x, (b, h * w, -1))

        qkv = ops.reshape(self.qkv(x), (b, h * w, 3, self.num_heads, -1))
        qkv = ops.transpose(qkv, (2, 0, 3, 1, 4))

        qkv = ops.reshape(qkv, (3, b * self.num_heads, h * w, -1))
        q, k, v = ops.unstack(qkv, axis=0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.use_rel_pos:
            attn_bias = get_decomposed_rel_pos(q, self.rel_pos_h, self.rel_pos_w, (h, w), (h, w))
        else:
            attn_bias = None

        q = ops.mul(q, self.scale)
        attn = self.q_matmul_k(q, k)

        if attn_bias is not None:
            attn = attn + attn_bias
        attn = ops.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        out = self.attn_matmul_v(attn, v)
        out = ops.reshape(out, (b, self.num_heads, h * w, -1))
        out = ops.transpose(out, (0, 2, 1, 3))
        out = ops.reshape(out, (b, h * w, -1))
        out = self.proj(out)
        out = ops.reshape(out, (b, h, w, -1))
        return out


def get_rel_pos(q_size: int, k_size: int, rel_pos: ms.Tensor) -> ms.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = ops.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = ops.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = ops.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def get_decomposed_rel_pos(
    q: ms.Tensor,
    rel_pos_h: ms.Tensor,
    rel_pos_w: ms.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> ms.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    rh = get_rel_pos(q_h, k_h, rel_pos_h)
    rw = get_rel_pos(q_w, k_w, rel_pos_w)

    b, _, dim = q.shape
    r_q = q.reshape(b, q_h, q_w, dim)
    dtype = r_q.dtype
    # rel_h = ops.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_h = ops.BatchMatMul(transpose_b=True)(r_q, ops.unsqueeze(rh, 0).astype(dtype).repeat(b, axis=0))
    # rel_w = ops.einsum("bhwc,wkc->bhwk", r_q, Rw)
    rel_w = ops.mul(ops.unsqueeze(r_q, -2), ops.unsqueeze(ops.unsqueeze(rw, 0), 0).astype(dtype)).sum(axis=-1)

    attn_bias = rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    attn_bias = ops.reshape(attn_bias, (-1, q_h * q_w, k_h * k_w))
    return attn_bias


class Block(nn.Cell):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        proj_drop: float = 0.,
        attn_drop: float = 0.,
        init_values: Optional[float] = None,
        drop_path: float = 0.,
        norm_layer: nn.Cell = nn.LayerNorm,
        act_layer: nn.Cell = nn.GELU,
        mlp_layer: Callable = Mlp,
        use_rel_pos: bool = False,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()
        self.window_size = window_size
        self.norm1 = norm_layer((dim,))
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            norm_layer=norm_layer,
            use_rel_pos=use_rel_pos,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )
        self.ls1 = LayerScale(dim=dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer((dim,))
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )

        self.ls2 = LayerScale(dim=dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        b, h, w, _ = x.shape
        shortcut = x
        x = self.norm1(x)

        if self.window_size > 0:
            # window partition
            x, pad_hw = window_partition(x, self.window_size)
            x = self.drop_path1(self.ls1(self.attn(x)))
            # reverse window partition
            x = window_unpartition(x, self.window_size, pad_hw, (h, w))

        else:
            x = self.drop_path1(self.ls1(self.attn(x)))

        x = shortcut + x
        x = ops.reshape(x, (b, h * w, -1))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        x = ops.reshape(x, (b, h, w, -1))

        return x


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

    def __init__(self, dim: int, init_values: float = 1e-5):
        super(LayerScale, self).__init__()
        self.gamma = Parameter(initializer(init_values, dim))

    def construct(self, x):
        return self.gamma * x


def window_partition(x: ms.Tensor, window_size: int) -> Tuple[ms.Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [b, h, w, c].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [b * num_windows, window_size, window_size, c].
        (hp, Wp): padded height and width before partition
    """
    b, h, w, c = x.shape

    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size

    if pad_h > 0 or pad_w > 0:
        # replace ops.pad with ops.concat for better performance
        pad_mat1 = ops.zeros((b, h, pad_w, c), x.dtype)
        pad_mat2 = ops.zeros((b, pad_h, w + pad_w, c), x.dtype)
        x = ops.concat([ops.concat([x, pad_mat1], axis=2), pad_mat2], axis=1)

    hp, wp = h + pad_h, w + pad_w
    x = x.view(b, hp // window_size, window_size, wp // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).view(-1, window_size, window_size, c)

    return windows, (hp, wp)


def window_unpartition(
    windows: ms.Tensor, window_size: int, pad_hw: Optional[Tuple[int, int]], hw: Tuple[int, int],
) -> ms.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [b * num_windows, window_size, window_size, c].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (hp, wp).
        hw (Tuple): original height and width (h, w) before padding.

    Returns:
        x: unpartitioned sequences with [b, h, w, c].
    """
    hp, wp = pad_hw if pad_hw is not None else hw
    h, w = hw
    b = windows.shape[0] // (hp * wp // window_size // window_size)
    x = windows.view(b, hp // window_size, wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).view(b, hp, wp, -1)

    if hp > h or wp > w:
        x = x[:, :h, :w, :]
    return x


class VitSAM(nn.Cell):
    """Vision Transformer for Segment-Anything Model(SAM)"""

    def __init__(
        self,
        image_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 768,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        init_values: Optional[float] = None,
        pre_norm: bool = False,
        drop_rate: float = 0.,
        pos_drop_rate: float = 0.,
        proj_drop_rate: float = 0.,
        patch_drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.,
        embed_layer: Callable = partial(PatchEmbed, output_fmt=Format.NHWC, strict_img_size=False),
        norm_layer: Optional[Callable] = nn.LayerNorm,
        act_layer: Optional[Callable] = nn.GELU,
        block_fn: Callable = Block,
        mlp_layer: Callable = Mlp,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
        neck_chans: int = 256,
        global_pool: str = 'avg',
        head_hidden_size: Optional[int] = None,
    ) -> None:
        """
        Args:
            image_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            init_values: Layer-scale init values (layer-scale enabled if not None).
            drop_rate: Head dropout rate.
            pos_drop_rate: Position embedding dropout rate.
            attn_drop_rate: Attention dropout rate.
            drop_path_rate: Stochastic depth rate.
            weight_init: Weight initialization scheme.
            embed_layer: Patch embedding layer.
            norm_layer (nn.Cell): Normalization layer.
            act_layer (nn.Cell): Activation layer.
            block_fn: Transformer block layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes: Indexes for blocks using global attention. Used when window_size > 0.
            global_pool: Global pooling type.
            head_hidden_size: If set, use NormMlpHead
        """
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, epsilon=1e-6)
        act_layer = act_layer or nn.GELU
        self.num_classes = num_classes
        self.global_pool = global_pool
        # num_features for consistency with other models
        self.num_features = self.embed_dim = embed_dim

        self.image_size = image_size
        self.patch_embed = embed_layer(
            image_size=image_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,
        )
        grid_size = self.patch_embed.grid_size
        self.pos_embed: Optional[ms.Parameter] = None
        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = ms.Parameter(ops.zeros((1, grid_size[0], grid_size[1], embed_dim)))
        self.pos_drop = Dropout(p=pos_drop_rate)
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
                use_rel_pos=use_rel_pos, window_size=window_size if i not in global_attn_indexes else 0,
                input_size=grid_size,
            ) for i in range(depth)
        ])

        if neck_chans:
            self.neck = nn.SequentialCell(
                nn.Conv2d(
                    embed_dim,
                    neck_chans,
                    kernel_size=1,
                    has_bias=False,
                ),
                LayerNorm2d(neck_chans),
                nn.Conv2d(neck_chans, neck_chans, kernel_size=3, padding=1, has_bias=False, pad_mode='pad'),
                LayerNorm2d(neck_chans),
            )
            self.num_features = neck_chans
        else:
            if head_hidden_size:
                self.neck = nn.Identity()
            else:
                # should have a final norm with standard ClassifierHead
                self.neck = LayerNorm2d(embed_dim)
            neck_chans = embed_dim

        self.head = ClassifierHead(
            neck_chans,
            num_classes,
            hidden_size=head_hidden_size,
            pool_type=global_pool,
            drop_rate=drop_rate,
        )

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.neck(x.permute(0, 3, 1, 2))
        return x

    def forward_head(self, x, pre_logits: bool = False):
        return self.head(x, pre_logits=True) if pre_logits else self.head(x)

    def construct(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


class ClassifierHead(nn.Cell):
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_size: Optional[int] = None,
        pool_type: str = 'avg',
        drop_rate: float = 0.,
        act_layer: Optional[nn.Cell] = nn.Tanh
    ):
        super().__init__()
        assert pool_type == "avg", "only support avg pooling"
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.num_features = in_features
        keep_dims = True if hidden_size else False
        self.global_pool = GlobalAvgPooling(keep_dims=keep_dims)
        self.norm = LayerNorm2d(in_features) if hidden_size else nn.Identity()
        self.flatten = nn.Flatten() if hidden_size else nn.Identity()

        if hidden_size:
            self.pre_logits = nn.SequentialCell(OrderedDict([
                ('fc', nn.Dense(in_features, hidden_size)),
                ('act', act_layer()),
            ]))
            self.num_features = hidden_size
        else:
            self.pre_logits = nn.Identity()
        self.drop = Dropout(drop_rate)
        self.fc = nn.Dense(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def construct(self, x, pre_logits: bool = False):
        x = self.global_pool(x)
        x = self.norm(x)
        x = self.flatten(x)
        x = self.pre_logits(x)
        x = self.drop(x)
        if pre_logits:
            return x
        x = self.fc(x)
        return x


class LayerNorm2d(nn.Cell):
    def __init__(self, num_channels: int, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.weight = ms.Parameter(ops.ones(num_channels))
        self.bias = ms.Parameter(ops.zeros(num_channels))
        self.eps = epsilon

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        u = x.mean(1, keep_dims=True)
        s = (x - u).pow(2).mean(1, keep_dims=True)
        x = (x - u) / ops.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


@register_model
def samvit_base_patch16(pretrained: bool = False, num_classes: int = 0, in_channels: int = 3, **kwargs):
    default_cfg = default_cfgs["samvit_base_patch16"]
    model = VitSAM(
        image_size=1024, patch_size=16, embed_dim=768, depth=12, num_heads=12, global_attn_indexes=[2, 5, 8, 11],
        window_size=14, use_rel_pos=True, num_classes=num_classes, in_chans=in_channels, **kwargs
    )

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels, **kwargs)
    return model


@register_model
def samvit_large_patch16(pretrained: bool = False, num_classes: int = 0, in_channels: int = 3, **kwargs):
    default_cfg = default_cfgs["samvit_large_patch16"]
    model = VitSAM(
        image_size=1024, patch_size=16, embed_dim=1024, depth=24, num_heads=16, global_attn_indexes=[5, 11, 17, 23],
        window_size=14, use_rel_pos=True, num_classes=num_classes, in_chans=in_channels, **kwargs
    )

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels, **kwargs)
    return model


@register_model
def samvit_huge_patch16(pretrained: bool = False, num_classes: int = 0, in_channels: int = 3, **kwargs):
    default_cfg = default_cfgs["samvit_large_patch16"]
    model = VitSAM(
        image_size=1024, patch_size=16, embed_dim=1280, depth=32, num_heads=16, global_attn_indexes=[7, 15, 23, 31],
        window_size=14, use_rel_pos=True, num_classes=num_classes, in_chans=in_channels, **kwargs,
    )

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels, **kwargs)
    return model
