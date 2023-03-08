# Copyright IBM All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Mostly copy-paste from https://github.com/yitu-opensource/T2T-ViT/blob/main/models/token_transformer.py
"""

import math
import numpy as np
from mindspore import nn
from mindspore import ops
from mindspore import Tensor
from mindspore.numpy import ones
import collections.abc
from itertools import repeat


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


class DropPath(nn.Cell):
    """ DropPath (Stochastic Depth) regularization layers """

    def __init__(self,
                 drop_prob: float = 0.,
                 scale_by_keep: bool = True) -> None:
        super().__init__()
        self.keep_prob = 1. - drop_prob
        self.scale_by_keep = scale_by_keep
        self.dropout = nn.Dropout(self.keep_prob)

    def construct(self, x: Tensor) -> Tensor:
        if self.keep_prob == 1. or not self.training:
            return x
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = self.dropout(ones(shape))
        if not self.scale_by_keep:
            random_tensor = ops.mul(random_tensor, self.keep_prob)
        return x * random_tensor


def get_sinusoid_encoding(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return ops.FloatTensor(sinusoid_table).unsqueeze(0)


class Token_performer(nn.Cell):
    def __init__(self, dim, in_dim, head_cnt=1, kernel_ratio=0.5, dp1=0.1, dp2=0.1):
        # def __init__(self, dim, in_dim, head_cnt=1, kernel_ratio=0.5, dp1=0.0, dp2=0.0):
        super().__init__()
        self.emb = in_dim * head_cnt  # we use 1, so it is no need here
        self.kqv = nn.Dense(dim, 3 * self.emb)
        self.dp = nn.Dropout(dp1)
        self.proj = nn.Dense(self.emb, self.emb)
        self.head_cnt = head_cnt
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(self.emb)
        self.epsilon = 1e-8  # for stable in division

        self.mlp = nn.Sequential(
            nn.Dense(self.emb, 1 * self.emb),
            nn.GELU(),
            nn.Dense(1 * self.emb, self.emb),
            nn.Dropout(dp2),
        )

        self.m = int(self.emb * kernel_ratio)
        self.w = ops.randn(self.m, self.emb)
        self.w = nn.Parameter(nn.init.orthogonal_(self.w) * math.sqrt(self.m), requires_grad=False)

    def prm_exp(self, x):
        xd = ((x * x).sum(dim=-1, keepdim=True)).repeat(1, 1, self.m) / 2
        wtx = ops.einsum('bti,mi->btm', x.float(), self.w)

        return ops.exp(wtx - xd) / math.sqrt(self.m)

    def single_attn(self, x):
        k, q, v = ops.split(self.kqv(x), self.emb, dim=-1)
        kp, qp = self.prm_exp(k), self.prm_exp(q)  # (B, T, m), (B, T, m)
        D = ops.einsum('bti,bi->bt', qp, kp.sum(dim=1)).unsqueeze(dim=2)  # (B, T, m) * (B, m) -> (B, T, 1)
        kptv = ops.einsum('bin,bim->bnm', v.float(), kp)  # (B, emb, m)
        y = ops.einsum('bti,bni->btn', qp, kptv) / (D.repeat(1, 1, self.emb) + self.epsilon)  # (B, T, emb)/Diag
        # skip connection
        y = v + self.dp(self.proj(y))  # same as token_transformer in T2T layer, use v as skip connection

        return y

    def forward(self, x):
        x = self.single_attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Mlp(nn.Cell):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Dense(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Dense(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Cell):
    def __init__(self, dim, num_heads=8, in_dim=None, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.in_dim = in_dim
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Dense(dim, in_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Dense(in_dim, in_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.in_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.in_dim)
        x = self.proj(x)
        x = self.proj_drop(x)

        # skip connection
        x = v.squeeze(1) + x  # because the original x has different size with current x, use v to do skip connection

        return x


class Token_transformer(nn.Cell):

    def __init__(self, dim, in_dim, num_heads, mlp_ratio=1., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, in_dim=in_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
            proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(in_dim)
        self.mlp = Mlp(in_features=in_dim, hidden_features=int(in_dim * mlp_ratio), out_features=in_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = self.attn(self.norm1(x))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class T2T(nn.Cell):
    """
    Tokens-to-Token encoding Cell
    """

    def __init__(self, img_size=224, patch_size=16, tokens_type='transformer', in_chans=3, embed_dim=768, token_dim=64):
        super().__init__()

        if patch_size == 12:
            kernel_size = ((7, 4, 2), (3, 3, 1), (3, 1, 1))
        elif patch_size == 16:
            kernel_size = ((7, 4, 2), (3, 2, 1), (3, 2, 1))
        else:
            raise ValueError(f"Unknown patch size {patch_size}")

        self.soft_split0 = nn.Unfold(kernel_size=to_2tuple(kernel_size[0][0]), stride=to_2tuple(kernel_size[0][1]),
                                     padding=to_2tuple(kernel_size[0][2]))
        self.soft_split1 = nn.Unfold(kernel_size=to_2tuple(kernel_size[1][0]), stride=to_2tuple(kernel_size[1][1]),
                                     padding=to_2tuple(kernel_size[1][2]))
        self.soft_split2 = nn.Unfold(kernel_size=to_2tuple(kernel_size[2][0]), stride=to_2tuple(kernel_size[2][1]),
                                     padding=to_2tuple(kernel_size[2][2]))

        if tokens_type == 'transformer':
            # print('adopt transformer encoder for tokens-to-token')

            self.attention1 = Token_transformer(dim=in_chans * (kernel_size[0][0] ** 2), in_dim=token_dim, num_heads=1,
                                                mlp_ratio=1.0)
            self.attention2 = Token_transformer(dim=token_dim * (kernel_size[1][0] ** 2), in_dim=token_dim, num_heads=1,
                                                mlp_ratio=1.0)
            self.project = nn.Dense(token_dim * (kernel_size[2][0] ** 2), embed_dim)

        elif tokens_type == 'performer':

            self.attention1 = Token_performer(dim=in_chans * (kernel_size[0][0] ** 2), in_dim=token_dim,
                                              kernel_ratio=0.5)
            self.attention2 = Token_performer(dim=token_dim * (kernel_size[1][0] ** 2), in_dim=token_dim,
                                              kernel_ratio=0.5)
            self.project = nn.Dense(token_dim * (kernel_size[2][0] ** 2), embed_dim)

        self.num_patches = (img_size // (kernel_size[0][1] * kernel_size[1][1] * kernel_size[2][1])) * (img_size // (
                kernel_size[0][1] * kernel_size[1][1] * kernel_size[2][
            1]))  # there are 3 sfot split, stride are 4,2,2 seperately

    def forward(self, x):
        # step0: soft split
        x = self.soft_split0(x).transpose(1, 2)

        # iteration1: re-structurization/reconstruction
        x = self.attention1(x)
        B, new_HW, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        # iteration1: soft split
        x = self.soft_split1(x).transpose(1, 2)

        # iteration2: re-structurization/reconstruction
        x = self.attention2(x)
        B, new_HW, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        # iteration2: soft split
        x = self.soft_split2(x).transpose(1, 2)

        # final tokens
        x = self.project(x)

        return x


class SharedT2T(nn.Cell):
    """
    Tokens-to-Token encoding Cell
    """

    def __init__(self, img_size=224, patch_size=16, tokens_type='transformer', in_chans=3, embed_dim=768, token_dim=64):
        super().__init__()

        if patch_size == 12:
            kernel_size = ((7, 4, 2), (3, 3, 1), (3, 1, 1))
        elif patch_size == 16:
            kernel_size = ((7, 4, 2), (3, 2, 1), (3, 2, 1))
        else:
            raise ValueError(f"Unknown patch size {patch_size}")

        if tokens_type == 'transformer':
            # print('adopt transformer encoder for tokens-to-token')
            self.soft_split0 = nn.Unfold(kernel_size=to_2tuple(kernel_size[0][0]), stride=to_2tuple(kernel_size[0][1]),
                                         padding=to_2tuple(kernel_size[0][2]))
            self.soft_split1 = nn.Unfold(kernel_size=to_2tuple(kernel_size[1][0]), stride=to_2tuple(kernel_size[1][1]),
                                         padding=to_2tuple(kernel_size[1][2]))
            self.soft_split2 = nn.Unfold(kernel_size=to_2tuple(kernel_size[2][0]), stride=to_2tuple(kernel_size[2][1]),
                                         padding=to_2tuple(kernel_size[2][2]))

            self.attention1 = Token_transformer(dim=in_chans * (kernel_size[0][0] ** 2), in_dim=token_dim, num_heads=1,
                                                mlp_ratio=1.0)
            self.attention2 = Token_transformer(dim=token_dim * (kernel_size[1][0] ** 2), in_dim=token_dim, num_heads=1,
                                                mlp_ratio=1.0)
            self.project = nn.Dense(token_dim * (kernel_size[2][0] ** 2), embed_dim)

        self.num_patches = (img_size // (kernel_size[0][1] * kernel_size[1][1] * kernel_size[2][1])) * (img_size // (
                kernel_size[0][1] * kernel_size[1][1] * kernel_size[2][
            1]))  # there are 3 sfot split, stride are 4,2,2 seperately

    def forward(self, x):
        # step0: soft split
        x = self.soft_split0(x).transpose(1, 2)

        # iteration1: re-structurization/reconstruction
        x = self.attention1(x)
        B, new_HW, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        # iteration1: soft split
        x = self.soft_split1(x).transpose(1, 2)

        # iteration2: re-structurization/reconstruction
        x = self.attention2(x)
        B, new_HW, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        # iteration2: soft split
        x = self.soft_split2(x).transpose(1, 2)

        # final tokens
        x = self.project(x)

        return x
