"""
MindSpore implementation of `SwinTransformer V2`.
Refer to Swin Transformer V2: Scaling Up Capacity and Resolution.
"""

from typing import List, Optional, Tuple, Union

import numpy as np

import mindspore as ms
import mindspore.common.initializer as init
from mindspore import Parameter, Tensor
from mindspore import dtype as mstype
from mindspore import nn, ops

from .helpers import _ntuple, load_pretrained
from .layers import DropPath, Identity
from .layers.compatibility import Dropout
from .registry import register_model

__all__ = [
    "SwinTransformerV2",
    "swinv2_tiny_window8",
    "swinv2_tiny_window16",
    "swinv2_small_window8",
    "swinv2_small_window16",
    "swinv2_base_window8",
    "swinv2_base_window16",
]


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "first_conv": "patch_embed.proj",
        "classifier": "head",
        **kwargs,
    }


default_cfgs = {
    "swinv2_tiny_window8": _cfg(
        url="https://download.mindspore.cn/toolkits/mindcv/swinv2/swinv2_tiny_window8-3ef8b787.ckpt"
    ),
    "swinv2_tiny_window16": _cfg(url=""),
    "swinv2_small_window8": _cfg(url=""),
    "swinv2_small_window16": _cfg(url=""),
    "swinv2_base_window8": _cfg(url=""),
    "swinv2_base_window16": _cfg(url=""),
}


to_2tuple = _ntuple(2)


class Roll(nn.Cell):
    def __init__(self, shift_size: int, shift_axis: Tuple[int, int] = (1, 2)) -> None:
        super(Roll, self).__init__()
        self.shift_size = to_2tuple(shift_size)
        self.shift_axis = shift_axis

    def construct(self, x: Tensor) -> Tensor:
        x = ms.numpy.roll(x, self.shift_size, self.shift_axis)
        return x


class WindowPartition(nn.Cell):
    def __init__(self, window_size: int) -> None:
        super().__init__()
        self.window_size = window_size

    def construct(self, x: Tensor) -> Tensor:
        b, h, w, c = x.shape
        x = x.reshape(b, h // self.window_size, self.window_size, w // self.window_size, self.window_size, c)
        x = x.transpose(0, 1, 3, 2, 4, 5)
        x = x.reshape(b * h * w // (self.window_size**2), self.window_size, self.window_size, c)

        return x


def window_partition(x, window_size):
    b, h, w, c = x.shape
    x = x.reshape(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.transpose(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, c)
    return windows


class WindowReverse(nn.Cell):
    def __init__(self) -> None:
        super().__init__()

    def construct(self, windows: Tensor, window_size: int, h: int, w: int) -> Tensor:
        b = windows.shape[0] // (h * w // window_size // window_size)
        x = windows.reshape(b, h // window_size, w // window_size, window_size, window_size, -1)
        x = x.transpose(0, 1, 3, 2, 4, 5)
        x = x.reshape(b, h, w, -1)
        return x


class LogSpacedCPB(nn.Cell):
    def __init__(
        self,
        window_size: Tuple[int, int],
        num_heads: int,
        pretrained_window_size: Tuple[int, int],
    ) -> None:
        super().__init__()
        self.window_size = window_size  # Wh, Ww
        # mlp to generate continuous relative position bias
        self.num_heads = num_heads
        self.cpb_mlp0 = nn.Dense(2, 512, has_bias=True)
        self.cpb_act1 = nn.ReLU()
        self.cpb_mlp2 = nn.Dense(512, num_heads, has_bias=False)

        relative_coords_h = np.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=float)
        relative_coords_w = np.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=float)
        relative_coords_table = np.stack(np.meshgrid(relative_coords_h, relative_coords_w, indexing="ij"), axis=0)
        relative_coords_table = np.transpose(relative_coords_table, (1, 2, 0))
        relative_coords_table = np.expand_dims(relative_coords_table, axis=0)
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= pretrained_window_size[0] - 1
            relative_coords_table[:, :, :, 1] /= pretrained_window_size[1] - 1
        else:
            relative_coords_table[:, :, :, 0] /= self.window_size[0] - 1
            relative_coords_table[:, :, :, 1] /= self.window_size[1] - 1
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = (
                np.sign(relative_coords_table) * np.log2(np.abs(relative_coords_table) + 1) / np.log2(8)
        )

        self.relative_coords_table = Parameter(
            Tensor(relative_coords_table, mstype.float32), requires_grad=False
        )

        # get pair-wise relative position index for each token inside the window
        coords_h = np.arange(window_size[0])
        coords_w = np.arange(window_size[1])
        coords = np.stack(np.meshgrid(coords_h, coords_w, indexing="ij"), axis=0)  # 2, Wh, Ww
        coords_flatten = coords.reshape(coords.shape[0], -1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = np.transpose(relative_coords, (1, 2, 0))  # Wh*Ww, Wh*Ww, 2

        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1

        relative_position_index = np.sum(relative_coords, axis=-1)  # Wh*Ww, Wh*Ww

        self.relative_position_index = Parameter(
            Tensor(relative_position_index, mstype.int32), requires_grad=False
        )

        self.sigmoid = ops.Sigmoid()

    def construct(self) -> Tensor:
        x = self.cpb_mlp0(self.relative_coords_table)
        x = self.cpb_act1(x)
        x = self.cpb_mlp2(x)
        x = x.reshape(-1, self.num_heads)
        relative_position_bias = x[ops.reshape(self.relative_position_index, (-1,))]
        relative_position_bias = ops.reshape(relative_position_bias, (
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1))
        relative_position_bias = relative_position_bias.transpose(2, 0, 1)
        relative_position_bias = 16 * self.sigmoid(relative_position_bias)

        relative_position_bias = ops.expand_dims(relative_position_bias, axis=0)
        return relative_position_bias


class WindowCosineAttention(nn.Cell):
    def __init__(
        self,
        dim: Union[Tuple[int], int],
        window_size: Tuple[int, int],
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        pretrained_window_size: Tuple[int, int] = (0, 0),
    ):
        super(WindowCosineAttention, self).__init__()
        if isinstance(dim, tuple) and len(dim) == 1:
            dim = dim[0]
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads

        self.matmul = ops.BatchMatMul()

        logit_scale = Tensor((ops.log(10 * ops.ones((num_heads, 1, 1), mstype.float32))))
        self.logit_scale = Parameter(logit_scale, requires_grad=True)

        max = Tensor(100, mstype.float32)
        self.value_max = ops.log(max)
        self.value_min = Tensor((-1000), mstype.float32)

        self.q = nn.Dense(in_channels=dim, out_channels=dim, has_bias=qkv_bias)
        self.k = nn.Dense(in_channels=dim, out_channels=dim, has_bias=False)
        self.v = nn.Dense(in_channels=dim, out_channels=dim, has_bias=qkv_bias)

        self.Normalize = ops.L2Normalize(axis=-1, epsilon=1e-12)

        # get pair-wise relative position index for each token inside the window
        self.relative_position_bias = LogSpacedCPB(self.window_size, num_heads, pretrained_window_size)

        self.attn_drop = Dropout(p=attn_drop)
        self.proj = nn.Dense(in_channels=dim, out_channels=dim, has_bias=True)
        self.softmax = nn.Softmax(axis=-1)
        self.proj_drop = Dropout(p=proj_drop)

    def construct(self, x: Tensor, mask=None) -> Tensor:
        B_, N, C = x.shape

        q = ops.reshape(self.q(x), (B_, N, self.num_heads, C // self.num_heads))
        q = self.Normalize(q)
        q = ops.transpose(q, (0, 2, 1, 3))

        k = ops.reshape(self.k(x), (B_, N, self.num_heads, C // self.num_heads))
        k = self.Normalize(k)
        k = ops.transpose(k, (0, 2, 3, 1))

        v = ops.reshape(self.v(x), (B_, N, self.num_heads, C // self.num_heads))
        v = ops.transpose(v, (0, 2, 1, 3))

        attn = self.matmul(q, k)

        logit_scale = ops.clip_by_value(self.logit_scale, clip_value_min=self.value_min, clip_value_max=self.value_max)
        logit_scale = ops.exp(logit_scale)

        attn = attn * logit_scale

        attn = attn + self.relative_position_bias()

        if mask is not None:
            nW, ws2, _ = mask.shape
            mask = ops.reshape(mask, (1, -1, 1, ws2, ws2))
            attn = ops.reshape(attn, (B_ // nW, nW, self.num_heads, N, N,)) + mask
            attn = ops.reshape(attn, (-1, self.num_heads, N, N,))
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = ops.reshape(ops.transpose(self.matmul(attn, v), (0, 2, 1, 3)), (B_, N, C))
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Mlp(nn.Cell):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Optional[nn.Cell] = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Dense(in_channels=in_features, out_channels=hidden_features, has_bias=True)
        self.act = act_layer()
        self.fc2 = nn.Dense(in_channels=hidden_features, out_channels=out_features, has_bias=True)
        self.drop = Dropout(p=drop)

    def construct(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwinTransformerBlock(nn.Cell):
    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: nn.Cell = nn.GELU,
        norm_layer: nn.Cell = nn.LayerNorm,
        pretrained_window_size: int = 0,
    ) -> None:
        super(SwinTransformerBlock, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        if isinstance(dim, int):
            dim = (dim,)

        self.norm1 = norm_layer(dim, epsilon=1e-6)

        self.attn = WindowCosineAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            pretrained_window_size=to_2tuple(pretrained_window_size))

        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        self.norm2 = norm_layer(dim, epsilon=1e-6)
        mlp_hidden_dim = int((dim[0] if isinstance(dim, tuple) else dim) * mlp_ratio)
        self.mlp = Mlp(in_features=dim[0] if isinstance(dim, tuple) else dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = np.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            # img_mask: [1, 56, 56, 1] window_size: 7
            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.reshape(-1, self.window_size * self.window_size)
            attn_mask = mask_windows[:, np.newaxis] - mask_windows[:, :, np.newaxis]
            # [64, 49, 49]
            attn_mask = Tensor(np.where(attn_mask == 0, 0.0, -100.0), dtype=mstype.float32)
            self.attn_mask = Parameter(attn_mask, requires_grad=False)
            self.roll_pos = Roll(self.shift_size)
            self.roll_neg = Roll(-self.shift_size)
        else:
            self.attn_mask = None

        self.window_partition = WindowPartition(self.window_size)
        self.window_reverse = WindowReverse()

    def construct(self, x: Tensor) -> Tensor:
        H, W = self.input_resolution
        B, _, C = x.shape

        shortcut = x
        # x = self.norm1(x)
        x = ops.reshape(x, (B, H, W, C,))

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = self.roll_neg(x)
        else:
            shifted_x = x

        # partition windows
        # nW*B, window_size, window_size, C
        x_windows = self.window_partition(shifted_x)
        # nW*B, window_size*window_size, C
        x_windows = ops.reshape(x_windows, (-1, self.window_size * self.window_size, C,))

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = ops.reshape(attn_windows, (-1, self.window_size, self.window_size, C,))
        shifted_x = self.window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = self.roll_pos(shifted_x)
        else:
            x = shifted_x

        x = ops.reshape(x, (B, H * W, C,))

        # FFN post-res-norm
        x = shortcut + self.drop_path(self.norm1(x))

        x = x + self.drop_path(self.norm2(self.mlp(x)))

        return x


class PatchMerging(nn.Cell):
    def __init__(
        self,
        input_resolution: Tuple[int, int],
        dim: int,
        norm_layer: nn.Cell = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim[0] if isinstance(dim, tuple) and len(dim) == 1 else dim
        # Default False
        self.reduction = nn.Dense(in_channels=4 * dim, out_channels=2 * dim, has_bias=False)
        self.norm = norm_layer([dim * 2], epsilon=1e-5)
        self.H, self.W = self.input_resolution
        self.H_2, self.W_2 = self.H // 2, self.W // 2
        self.H2W2 = int(self.H * self.W // 4)
        self.dim_mul_4 = int(dim * 4)
        self.H2W2 = int(self.H * self.W // 4)

    def construct(self, x: Tensor) -> Tensor:
        B = x.shape[0]
        x = ops.reshape(x, (B, self.H_2, 2, self.W_2, 2, self.dim))
        x = ops.transpose(x, (0, 1, 3, 4, 2, 5))
        x = ops.reshape(x, (B, self.H2W2, self.dim_mul_4))
        x = self.reduction(x)
        x = self.norm(x)

        return x


class BasicLayer(nn.Cell):
    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: List[float] = 0.0,
        norm_layer: nn.Cell = nn.LayerNorm,
        downsample: Optional[nn.Cell] = None,
        pretrained_window_size: int = 0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # build blocks
        self.blocks = nn.CellList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 pretrained_window_size=pretrained_window_size)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def construct(self, x: Tensor) -> Tensor:
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class PatchEmbed(nn.Cell):
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 96,
        norm_layer: Optional[nn.Cell] = None,
    ) -> None:
        super().__init__()
        image_size = to_2tuple(image_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [image_size[0] // patch_size[0], image_size[1] // patch_size[1]]
        self.image_size = image_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_channels=in_chans, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size,
                              pad_mode='pad', has_bias=True)

        if norm_layer is not None:
            if isinstance(embed_dim, int):
                embed_dim = (embed_dim,)
            self.norm = norm_layer(embed_dim, epsilon=1e-6)
        else:
            self.norm = None

    def construct(self, x: Tensor) -> Tensor:
        B = x.shape[0]
        x = ops.reshape(self.proj(x), (B, self.embed_dim, -1))
        x = ops.transpose(x, (0, 2, 1))  # B Ph*Pw C

        if self.norm is not None:
            x = self.norm(x)
        return x


class SwinTransformerV2(nn.Cell):
    r"""SwinTransformerV2 model class, based on
    `"Swin Transformer V2: Scaling Up Capacity and Resolution" <https://arxiv.org/abs/2111.09883>`_

    Args:
        image_size: Input image size. Default: 256.
        patch_size: Patch size. Default: 4.
        in_channels: Number the channels of the input. Default: 3.
        num_classes: Number of classification classes. Default: 1000.
        embed_dim: Patch embedding dimension. Default: 96.
        depths: Depth of each Swin Transformer layer. Default: [2, 2, 6, 2].
        num_heads: Number of attention heads in different layers. Default: [3, 6, 12, 24].
        window_size: Window size. Default: 7.
        mlp_ratio: Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias: If True, add a bias for query, key, value. Default: True.
        drop_rate: Drop probability for the Dropout layer. Default: 0.
        attn_drop_rate: Attention drop probability for the Dropout layer. Default: 0.
        drop_path_rate: Stochastic depth rate. Default: 0.1.
        norm_layer: Normalization layer. Default: nn.LayerNorm.
        patch_norm: If True, add normalization after patch embedding. Default: True.
        pretrained_window_sizes: Pretrained window sizes of each layer. Default: [0, 0, 0, 0].
    """

    def __init__(
        self,
        image_size: int = 256,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 96,
        depths: List[int] = [2, 2, 6, 2],
        num_heads: List[int] = [3, 6, 12, 24],
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer: nn.Cell = nn.LayerNorm,
        patch_norm: bool = True,
        pretrained_window_sizes: List[int] = [0, 0, 0, 0],
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            image_size=image_size, patch_size=patch_size, in_chans=in_channels, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.pos_drop = Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x for x in np.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.CellList()
        self.final_seq = num_patches  # downsample seq_length
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                input_resolution=(patches_resolution[0] // (2**i_layer),
                                  patches_resolution[1] // (2**i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                pretrained_window_size=pretrained_window_sizes[i_layer]
            )
            # downsample seq_length
            if i_layer < self.num_layers - 1:
                self.final_seq = self.final_seq // 4
            self.layers.append(layer)
        self.head = nn.Dense(self.num_features, self.num_classes)

        self.norm = norm_layer([self.num_features, ], epsilon=1e-6)
        self.avgpool = ops.ReduceMean(keep_dims=False)

        self._initialize_weights()

    def _initialize_weights(self):
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(init.initializer(init.HeUniform(), cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(init.initializer("zeros", cell.bias.shape, cell.bias.dtype))
            elif isinstance(cell, nn.LayerNorm):
                cell.gamma.set_data(init.initializer("ones", cell.gamma.shape, cell.gamma.dtype))
                cell.beta.set_data(init.initializer("zeros", cell.beta.shape, cell.beta.dtype))
            elif isinstance(cell, nn.Dense):
                cell.weight.set_data(
                    init.initializer(init.TruncatedNormal(sigma=0.02), cell.weight.shape, cell.weight.dtype)
                )
                if cell.bias is not None:
                    cell.bias.set_data(init.initializer("zeros", cell.bias.shape, cell.bias.dtype))

    def forward_features(self, x: Tensor) -> Tensor:
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)  # B L C
        x = self.avgpool(ops.transpose(x, (0, 2, 1)), 2)  # B C 1
        return x

    def forward_head(self, x: Tensor) -> Tensor:
        x = self.head(x)
        return x

    def construct(self, x: Tensor) -> Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


@register_model
def swinv2_tiny_window8(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs["swinv2_tiny_window8"]
    model = SwinTransformerV2(in_channels=in_channels, num_classes=num_classes,
                              window_size=8, embed_dim=96, depths=[2, 2, 6, 2],
                              num_heads=[3, 6, 12, 24], **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def swinv2_tiny_window16(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs["swinv2_tiny_window16"]
    model = SwinTransformerV2(in_channels=in_channels, num_classes=num_classes,
                              window_size=16, embed_dim=96, depths=[2, 2, 6, 2],
                              num_heads=[3, 6, 12, 24], **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def swinv2_small_window8(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs["swinv2_small_window8"]
    model = SwinTransformerV2(in_channels=in_channels, num_classes=num_classes,
                              window_size=8, embed_dim=96, depths=[2, 2, 18, 2],
                              num_heads=[3, 6, 12, 24], **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def swinv2_small_window16(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs["swinv2_small_window16"]
    model = SwinTransformerV2(in_channels=in_channels, num_classes=num_classes,
                              window_size=16, embed_dim=96, depths=[2, 2, 18, 2],
                              num_heads=[3, 6, 12, 24], **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def swinv2_base_window8(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs["swinv2_base_window8"]
    model = SwinTransformerV2(in_channels=in_channels, num_classes=num_classes,
                              window_size=8, embed_dim=128, depths=[2, 2, 18, 2],
                              num_heads=[4, 8, 16, 32], **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def swinv2_base_window16(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs["swinv2_base_window16"]
    model = SwinTransformerV2(in_channels=in_channels, num_classes=num_classes,
                              window_size=16, embed_dim=128, depths=[2, 2, 18, 2],
                              num_heads=[4, 8, 16, 32], **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model
