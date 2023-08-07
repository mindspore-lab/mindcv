import math
from typing import Optional, Tuple

import numpy as np

import mindspore as ms
from mindspore import Parameter, Tensor, nn, ops
from mindspore.common.initializer import TruncatedNormal, initializer

from .layers.drop_path import DropPath
from .layers.mlp import Mlp
from .layers.patch_embed import PatchEmbed


class RelativePositionBiasWithCLS(nn.Cell):
    def __init__(
        self,
        window_size: Tuple[int],
        num_heads: int
    ):
        super(RelativePositionBiasWithCLS, self).__init__()
        self.window_size = window_size
        self.num_tokens = window_size[0] * window_size[1]

        num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        # 3: cls to token, token to cls, cls to cls
        self.relative_position_bias_table = Parameter(
            Tensor(np.zeros((num_relative_distance, num_heads)), dtype=ms.float16)
        )
        coords_h = np.arange(window_size[0]).reshape(window_size[0], 1).repeat(window_size[1], 1).reshape(1, -1)
        coords_w = np.arange(window_size[1]).reshape(1, window_size[1]).repeat(window_size[0], 0).reshape(1, -1)
        coords_flatten = np.concatenate([coords_h, coords_w], axis=0)  # [2, Wh * Ww]

        relative_coords = coords_flatten[:, :, np.newaxis] - coords_flatten[:, np.newaxis, :]  # [2, Wh * Ww, Wh * Ww]
        relative_coords = relative_coords.transpose(1, 2, 0)  # [Wh * Ww, Wh * Ww, 2]
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[0] - 1

        relative_position_index = np.zeros((self.num_tokens + 1, self.num_tokens + 1),
                                           dtype=relative_coords.dtype)  # [Wh * Ww + 1, Wh * Ww + 1]
        relative_position_index[1:, 1:] = relative_coords.sum(-1)
        relative_position_index[0, 0:] = num_relative_distance - 3
        relative_position_index[0:, 0] = num_relative_distance - 2
        relative_position_index[0, 0] = num_relative_distance - 1
        relative_position_index = Tensor(relative_position_index.reshape(-1))

        self.one_hot = nn.OneHot(axis=-1, depth=num_relative_distance, dtype=ms.float16)
        self.relative_position_index = Parameter(self.one_hot(relative_position_index), requires_grad=False)

    def construct(self):
        out = ops.matmul(self.relative_position_index, self.relative_position_bias_table)
        out = ops.reshape(out, (self.num_tokens + 1, self.num_tokens + 1, -1))
        out = ops.transpose(out, (2, 0, 1))
        out = ops.expand_dims(out, 0)
        return out


class Attention(nn.Cell):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        attn_head_dim: Optional[int] = None,
    ):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * num_heads

        if qk_scale:
            self.scale = Tensor(qk_scale)
        else:
            self.scale = Tensor(head_dim ** -0.5)

        self.qkv = nn.Dense(dim, all_head_dim * 3, has_bias=qkv_bias)

        self.attn_drop = nn.Dropout(1 - attn_drop)
        self.proj = nn.Dense(all_head_dim, dim)
        self.proj_drop = nn.Dropout(1 - proj_drop)

        self.mul = ops.Mul()
        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()
        self.unstack = ops.Unstack(axis=0)
        self.attn_matmul_v = ops.BatchMatMul()
        self.q_matmul_k = ops.BatchMatMul(transpose_b=True)

    def construct(self, x, rel_pos_bias=None):
        b, n, c = x.shape
        qkv = self.qkv(x)
        qkv = self.reshape(qkv, (b, n, 3, self.num_heads, c // self.num_heads))
        qkv = self.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = self.unstack(qkv)

        attn = self.q_matmul_k(q, k)
        attn = self.mul(attn, self.scale)

        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias

        attn = attn.astype(ms.float32)
        attn = ops.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        out = self.attn_matmul_v(attn, v)
        out = self.transpose(out, (0, 2, 1, 3))
        out = self.reshape(out, (b, n, c))
        out = self.proj(out)
        out = self.proj_drop(out)

        return out


class LayerScale(nn.Cell):
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
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        attn_head_dim: Optional[int] = None,
        mlp_ratio: float = 4.,
        drop_path: float = 0.,
        init_values: Optional[float] = None,
        act_layer: nn.Cell = nn.GELU,
        norm_layer: nn.Cell = nn.LayerNorm,
    ):
        super(Block, self).__init__()
        self.norm1 = norm_layer((dim,))
        self.attn = Attention(
            dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=proj_drop, attn_head_dim=attn_head_dim,
        )
        self.ls1 = LayerScale(dim=dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer((dim,))
        self.mlp = Mlp(
            in_features=dim, hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer, drop=proj_drop
        )
        self.ls2 = LayerScale(dim=dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def construct(self, x, rel_pos_bias=None):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), rel_pos_bias)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class VisionTransformerEncoder(nn.Cell):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        attn_head_dim: Optional[int] = None,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        pos_drop_rate: float = 0.,
        proj_drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.,
        init_values: Optional[float] = 0.1,
        act_layer: nn.Cell = nn.GELU,
        norm_layer: nn.Cell = nn.LayerNorm,
        use_abs_pos_emb: bool = False,
        use_rel_pos_bias: bool = False,
        use_shared_rel_pos_bias: bool = True,
        **kwargs
    ):
        super(VisionTransformerEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(image_size=img_size, patch_size=patch_size,
                                      in_chans=in_chans, embed_dim=embed_dim)
        self.num_patches = self.patch_embed.num_patches

        self.cls_token = Parameter(initializer(TruncatedNormal(0.02), (1, 1, embed_dim)))

        self.pos_embed = Parameter(
            initializer(TruncatedNormal(0.02), (1, self.num_patches + 1, embed_dim))) if use_abs_pos_emb else None
        self.pos_drop = nn.Dropout(1 - pos_drop_rate)

        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBiasWithCLS(
                window_size=self.patch_embed.patches_resolution, num_heads=num_heads)
        elif use_rel_pos_bias:
            self.rel_pos_bias = nn.CellList([
                RelativePositionBiasWithCLS(window_size=self.patch_embed.patches_resolution,
                                            num_heads=num_heads) for _ in range(depth)
            ])
        else:
            self.rel_pos_bias = None

        dpr = [x.item() for x in np.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.CellList([
            Block(
                dim=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop_rate, proj_drop=proj_drop_rate, attn_head_dim=attn_head_dim,
                mlp_ratio=mlp_ratio, drop_path=dpr[i], init_values=init_values,
                act_layer=act_layer, norm_layer=norm_layer
            ) for i in range(depth)
        ])

    def get_num_layers(self):
        return len(self.blocks)

    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def _init_weights(self):
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(
                    initializer(TruncatedNormal(0.02), cell.weight.shape, cell.weight.dtype)
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
            elif isinstance(cell, nn.Conv2d):
                cell.weight.set_data(
                    initializer(TruncatedNormal(0.02), cell.weight.shape, cell.weight.dtype)
                )
                if cell.bias is not None:
                    cell.bias.set_data(
                        initializer('zeros', cell.bias.shape, cell.bias.dtype)
                    )

    def _fix_init_weights(self):
        for i, block in enumerate(self.blocks):
            block.attn.proj.weight.set_data(
                ops.div(block.attn.proj.weight, math.sqrt(2.0 * (i + 1)))
            )
            block.mlp.fc2.weight.set_data(
                ops.div(block.mlp.fc2.weight, math.sqrt(2.0 * (i + 1)))
            )

    def forward_features(self, x):
        x = self.patch_embed(x)
        bsz = x.shape[0]

        cls_token = ops.broadcast_to(self.cls_token, (bsz, -1, -1))
        cls_token = cls_token.astype(x.dtype)
        x = ops.concat((cls_token, x), axis=1)

        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        if isinstance(self.rel_pos_bias, nn.CellList):
            for i, blk in enumerate(self.blocks):
                rel_pos_bias = self.rel_pos_bias[i]()
                x = blk(x, rel_pos_bias)
        else:
            rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
            for blk in self.blocks:
                x = blk(x, rel_pos_bias)

        return x

    def construct(self, x):
        return self.forward_features(x)
