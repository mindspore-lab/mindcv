"""NaViT"""

from typing import Callable

import numpy as np

import mindspore as ms
from mindspore import Parameter, Tensor, nn, ops
from mindspore.common.initializer import initializer

from .helpers import load_pretrained
from .layers.compatibility import Dropout
from .layers.drop_path import DropPath
from .layers.mlp import Mlp
from .registry import register_model

__all__ = [
    "NativeResolutionVisionTransformer",
    "navit_b_16_384",
]


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "classifier": "head.classifier",
        **kwargs,
    }


default_cfgs = {
    "navit_b_16_384": _cfg(url=""),
}


class RMSNorm(nn.Cell):
    """LayerNorm without bias and centering, norm at that last axis"""

    def __init__(
        self,
        shape,
        gamma_init="ones",
        epsilon=1e-5,
        dtype=ms.float32,
    ):
        """Initialize LayerNorm."""
        super(RMSNorm, self).__init__()
        self.epsilon = epsilon
        self.gamma = Parameter(initializer(gamma_init, shape, dtype=dtype), name="gamma")
        self.scale = Tensor(np.sqrt(shape[-1]), dtype=dtype)
        self.norm = ops.L2Normalize(axis=-1, epsilon=self.epsilon)

    def construct(self, input_x):
        x = self.norm(input_x)
        x = x * self.scale * self.gamma
        return x


# TODO: Flash Attention
class Attention(nn.Cell):
    """
    Attention layer implementation, Rearrange Input -> B x N x hidden size.

    Args:
        dim (int): The dimension of input features.
        num_heads (int): The number of attention heads. Default: 8.
        qkv_bias (bool): Specifies whether the linear layer uses a bias vector. Default: False.
        qk_norm (bool): Specifies whether to do normalization to q and k. Default: True.
        attn_drop (float): The drop rate of attention, greater than 0 and less equal than 1. Default: 0.0.
        proj_drop (float): The drop rate of output, greater than 0 and less equal than 1. Default: 0.0.
        norm_layer (nn.Cell): The normalization layer to apply. Default: RMSNorm

    Returns:
        Tensor, output tensor.

    Examples:
        >>> ops = Attention(768, 12)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Cell = RMSNorm,
    ):
        super(Attention, self).__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = Tensor(self.head_dim**-0.5)

        self.to_q = nn.Dense(dim, dim, has_bias=qkv_bias)
        self.to_k = nn.Dense(dim, dim, has_bias=qkv_bias)
        self.to_v = nn.Dense(dim, dim, has_bias=qkv_bias)

        self.q_norm = norm_layer((self.head_dim,)) if qk_norm else nn.Identity()
        self.k_norm = norm_layer((self.head_dim,)) if qk_norm else nn.Identity()

        self.attn_drop = Dropout(attn_drop)
        self.proj = nn.Dense(dim, dim)
        self.proj_drop = Dropout(proj_drop)

        self.attn_matmul_v = ops.BatchMatMul()
        self.q_matmul_k = ops.BatchMatMul(transpose_b=True)

    def rearange_in(self, x):
        b, n, _ = x.shape
        x = ops.reshape(x, (b, n, self.num_heads, -1))
        x = ops.transpose(x, (0, 2, 1, 3))
        return x

    def rearange_out(self, x):
        b, _, n, _ = x.shape
        x = ops.transpose(x, (0, 2, 1, 3))
        x = ops.reshape(x, (b, n, -1))
        return x

    def construct(self, x, context=None, token_mask=None):
        if context is None:
            context = x

        q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)

        q = self.rearange_in(q)
        k = self.rearange_in(k)
        v = self.rearange_in(v)

        q, k = self.q_norm(q), self.k_norm(k)

        attn = self.q_matmul_k(q, k)
        attn = ops.mul(attn, self.scale)

        # fp32 for softmax
        attn = attn.to(ms.float32)
        if token_mask is not None:
            token_mask = ops.unsqueeze(token_mask, 1)
            attn = ops.masked_fill(attn, ~token_mask, -ms.numpy.inf)
        attn = ops.softmax(attn, axis=-1).to(v.dtype)
        attn = self.attn_drop(attn)

        out = self.attn_matmul_v(attn, v)
        out = self.rearange_out(out)
        out = self.proj(out)
        out = self.proj_drop(out)

        return out


class Block(nn.Cell):
    """
    Transformer block implementation.

    Args:
        dim (int): The dimension of embedding.
        num_heads (int): The number of attention heads.
        qkv_bias (bool): Specifies whether the linear layer uses a bias vector. Default: False.
        attn_drop (float): The drop rate of attention, greater than 0 and less equal than 1. Default: 0.0.
        proj_drop (float): The drop rate of dense layer output, greater than 0 and less equal than 1. Default: 0.0.
        mlp_ratio (float): The ratio used to scale the input dimensions to obtain the dimensions of the hidden layer.
        drop_path (float): The drop rate for drop path. Default: 0.0.
        act_layer (nn.Cell): Activation function which will be stacked on top of the
            normalization layer (if not None), otherwise on top of the conv layer. Default: nn.GELU.
        norm_layer (nn.Cell): Norm layer that will be stacked on top of the convolution
            layer. Default: RMSNorm.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> ops = TransformerEncoder(768, 12, 12, 3072)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = True,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: nn.Cell = nn.GELU,
        norm_layer: nn.Cell = RMSNorm,
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
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer((dim,))
        self.mlp = mlp_layer(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=proj_drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def construct(self, x, token_mask=None):
        x = x + self.drop_path1(self.attn(self.norm1(x), token_mask=token_mask))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class NativeResolutionVisionTransformer(nn.Cell):
    def __init__(
        self,
        image_size: int = 384,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = True,
        drop_rate: float = 0.0,
        pos_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        pre_norm: bool = False,
        act_layer: nn.Cell = nn.GELU,
        norm_layer: nn.Cell = RMSNorm,
        mlp_layer: Callable = Mlp,
        block_fn: Callable = Block,
        num_classes: int = 1000,
        pool_type: str = "attn",
        max_num_each_group: int = 40,
    ):
        super().__init__()

        max_dim = image_size // patch_size
        patch_dim = in_channels * (patch_size**2)
        self.patch_embed = nn.SequentialCell(
            [norm_layer((patch_dim,)), nn.Dense(patch_dim, embed_dim), norm_layer((embed_dim,))]
        )

        self.pos_embed_height = nn.Embedding(max_dim, embed_dim)
        self.poe_embed_width = nn.Embedding(max_dim, embed_dim)
        self.pos_drop = Dropout(pos_drop_rate)

        self.norm_pre = norm_layer((embed_dim,)) if pre_norm else nn.Identity()
        dpr = np.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.CellList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    attn_drop=attn_drop_rate,
                    proj_drop=proj_drop_rate,
                    mlp_ratio=mlp_ratio,
                    drop_path=dpr[i],
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    mlp_layer=mlp_layer,
                )
                for i in range(depth)
            ]
        )

        self.head_drop = Dropout(drop_rate)
        self.pool_type = pool_type
        self.head = nn.Dense(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.max_num_each_group = max_num_each_group
        self.image_id_range = Tensor(np.arange(max_num_each_group), dtype=ms.int32)

        if self.pool_type == "attn":
            self.attn_pool_queries = Parameter(initializer("normal", (1, 1, embed_dim)))
            self.attn_pool = Attention(
                embed_dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                attn_drop=attn_drop_rate,
                proj_drop=proj_drop_rate,
                norm_layer=norm_layer,
            )

    def get_num_layers(self):
        return len(self.blocks)

    def no_weight_decay(self):
        return {"pos_embed_height", "pos_embed_width"}

    def _pos_embed(self, x, pos):
        h_inds, w_inds = ops.unbind(pos, dim=-1)

        h_pos = self.pos_embed_height(h_inds)
        w_pos = self.poe_embed_width(w_inds)

        x = x + h_pos + w_pos
        return self.pos_drop(x)

    def _token_mask(self, ind):
        mask = ind >= 0
        token_mask = ind[:, None, :] == ind[:, :, None]
        token_mask = ops.logical_and(token_mask, mask[:, None, :])
        return token_mask

    def _pool_mask(self, ind):
        mask = ind >= 0
        image_mask = ind[:, None, :] == self.image_id_range[None, :, None]
        pool_mask = ops.logical_and(image_mask, mask[:, None, :])
        return pool_mask

    def _avg_pool_with_mask(self, x, ind):
        pool_mask = self._pool_mask(ind)
        total = ops.BatchMatMul()(pool_mask.to(x.dtype), x)
        num = ops.sum(pool_mask, dim=-1, keepdim=True)
        num[num == 0] = 1
        return total / num

    def _attn_pool_with_mask(self, x, ind):
        pool_mask = self._pool_mask(ind)
        attn_pool_queries = ops.tile(self.attn_pool_queries, (x.shape[0], self.max_num_each_group, 1))
        x = self.attn_pool(attn_pool_queries, context=x, token_mask=pool_mask) + attn_pool_queries
        return x

    def forward_features(self, x, pos, ind):
        x = self.patch_embed(x)
        x = self._pos_embed(x, pos)
        x = self.norm_pre(x)
        token_mask = self._token_mask(ind)
        for block in self.blocks:
            x = block(x, token_mask=token_mask)
        return x

    def forward_head(self, x, ind):
        if self.pool_type == "avg":
            x = self._avg_pool_with_mask(x, ind)
        elif self.pool_type == "attn":
            x = self._attn_pool_with_mask(x, ind)

        x = self.head_drop(x)
        x = self.head(x)
        return x

    def construct(self, x, pos, ind):
        x = self.forward_features(x, pos, ind)
        x = self.forward_head(x, ind)
        return x


@register_model
def navit_b_16_384(pretrained: bool = False, num_classes: int = 1000, in_channels: int = 3, **kwargs):
    default_cfg = default_cfgs["navit_b_16_384"]
    model = NativeResolutionVisionTransformer(
        image_size=384,
        patch_size=16,
        in_channels=in_channels,
        embed_dim=768,
        depth=12,
        num_heads=12,
        num_classes=num_classes,
        **kwargs
    )
    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model
