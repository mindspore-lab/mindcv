"""Transformer in Transformer (TNT)"""
import math

import numpy as np
from scipy.stats import truncnorm

import mindspore.common.initializer as weight_init
import mindspore.nn as nn
import mindspore.ops.operations as P
from mindspore import Parameter, Tensor
from mindspore import dtype as mstype
from mindspore import ops

from .registry import register_model
from .utils import _ntuple, load_pretrained, make_divisible

__all__ = [
    "tnt_small",
    "tnt_base"
]


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000,
        'first_conv': 'patch_embed.proj',
        'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    "tnt_small": _cfg(
                url="https://storage.googleapis.com/huawei-mindspore-hk/TNT/tnt_s_patch16_224_ep138_acc_0.74.ckpt"),
    "tnt_base": _cfg(url="https://storage.googleapis.com/huawei-mindspore-hk/TNT/tnt_b_converted_0.795.ckpt")
}


class DropPath(nn.Cell):
    """
    Drop paths (Stochastic Depth) per sample
    (when applied in main path of residual blocks).

    Args:
        drop_prob(float): Probability of dropout
        ndim(int): Number of dimensions in input tensor

    Returns:
        Tensor: Output tensor after dropout
    """

    def __init__(self, drop_prob, ndim):
        super(DropPath, self).__init__()
        self.drop = nn.Dropout(keep_prob=1 - drop_prob)
        shape = (1,) + (1,) * (ndim + 1)
        self.ndim = ndim
        self.mask = Tensor(np.ones(shape), dtype=mstype.float32)

    def construct(self, *inputs, **kwargs):
        x = inputs[0]
        if not self.training:
            return x
        mask = ops.Tile()(self.mask, (x.shape[0],) + (1,) * (self.ndim + 1))
        out = self.drop(mask)
        out = out * x
        return out


class DropPath1D(DropPath):
    """DropPath1D"""

    def __init__(self, drop_prob):
        super(DropPath1D, self).__init__(drop_prob=drop_prob, ndim=1)


def trunc_array(shape, sigma=0.02):
    """output truncnormal array in shape"""
    return truncnorm.rvs(-2, 2, loc=0, scale=sigma, size=shape, random_state=None)


to_2tuple = _ntuple(2)


class UnfoldKernelEqPatch(nn.Cell):
    """
    UnfoldKernelEqPatch with better performance

    Args:
        kernel_size(tuple): kernel size (along each side)
        strides(tuple): Stride (along each side)

    Returns:
        Tensor, output tensor
    """

    def __init__(self, kernel_size, strides):
        super(UnfoldKernelEqPatch, self).__init__()
        assert kernel_size == strides
        self.kernel_size = kernel_size
        self.reshape = P.Reshape()
        self.transpose = P.Transpose()

    def construct(self, *inputs, **kwargs):
        inputs = inputs[0]
        b, c, h, w = inputs.shape
        inputs = self.reshape(inputs,
                              (b, c, h // self.kernel_size[0], self.kernel_size[0], w))
        inputs = self.transpose(inputs, (0, 2, 1, 3, 4))
        inputs = self.reshape(inputs, (-1, c, self.kernel_size[0], w // self.kernel_size[1], self.kernel_size[1]))
        inputs = self.transpose(inputs, (0, 3, 1, 2, 4))
        inputs = self.reshape(inputs, (-1, c, self.kernel_size[0], self.kernel_size[1]))
        # inputs = self.reshape(
        #     inputs,
        #     (B, C,
        #      H // self.kernel_size[0], self.kernel_size[0],
        #      W // self.kernel_size[1], self.kernel_size[1])
        # )
        # inputs = self.transpose(inputs, )

        return inputs


class PatchEmbed(nn.Cell):
    """
    Image to Visual Word Embedding

    Args:
        img_size(int): Image size (side, px)
        patch_size(int): Output patch size (side, px)
        in_chans(int): Number of input channels
        outer_dim(int): Number of output features (not used)
        inner_dim(int): Number of internal features
        inner_stride(int): Stride of patches (px)
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 outer_dim=768, inner_dim=24, inner_stride=4):
        super().__init__()
        _ = outer_dim
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.inner_dim = inner_dim
        self.num_words = math.ceil(patch_size[0] / inner_stride) * math.ceil(patch_size[1] / inner_stride)

        self.unfold = UnfoldKernelEqPatch(kernel_size=patch_size, strides=patch_size)
        # unfold_shape = [1, *patch_size, 1]
        # self.unfold = nn.Unfold(unfold_shape, unfold_shape, unfold_shape)
        self.proj = nn.Conv2d(in_channels=in_chans, out_channels=inner_dim, kernel_size=7, stride=inner_stride,
                              pad_mode='pad', padding=3, has_bias=True)

        self.reshape = P.Reshape()
        self.transpose = P.Transpose()

    def construct(self, *inputs, **kwargs):
        x = inputs[0]
        b, _ = x.shape[0], x.shape[1]
        x = self.unfold(x)  # B, Ck2, N
        x = self.proj(x)  # B*N, C, 8, 8
        x = self.reshape(x, (b * self.num_patches, self.inner_dim, -1,))  # B*N, 8*8, C
        x = self.transpose(x, (0, 2, 1))
        return x


class Attention(nn.Cell):
    """
    Attention layer

    Args:
        dim(int): Number of output features
        hidden_dim(int): Number of hidden features
        num_heads(int): Number of output heads
        qkv_bias(bool): Enable bias weights in Qk / v dense layers
        qk_scale(float): Qk scale (multiplier)
        attn_drop(float): Attention dropout rate
        proj_drop(float): Projection dropout rate
    """

    def __init__(self, dim, hidden_dim,
                 num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        head_dim = hidden_dim // num_heads
        self.head_dim = head_dim
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qk = nn.Dense(in_channels=dim, out_channels=hidden_dim * 2, has_bias=qkv_bias)
        # self.q = nn.Dense(in_channels=dim, out_channels=hidden_dim, has_bias=qkv_bias)
        # self.k = nn.Dense(in_channels=dim, out_channels=hidden_dim, has_bias=qkv_bias)
        self.v = nn.Dense(in_channels=dim, out_channels=dim, has_bias=qkv_bias)
        self.attn_drop = nn.Dropout(keep_prob=1.0 - attn_drop)
        self.proj = nn.Dense(in_channels=dim, out_channels=dim, has_bias=True)
        self.proj_drop = nn.Dropout(keep_prob=1.0 - proj_drop)
        self.softmax = nn.Softmax(axis=-1)
        self.matmul = P.BatchMatMul()

        self.reshape = P.Reshape()
        self.transpose = P.Transpose()

    def construct(self, *inputs, **kwargs):
        """Attention construct"""
        x = inputs[0]
        b, n, _ = x.shape
        qk = self.reshape(self.qk(x),
                          (b, n, 2, self.num_heads, self.head_dim))
        qk = self.transpose(qk, (2, 0, 3, 1, 4))
        q, k = qk[0], qk[1]

        v = self.reshape(self.v(x),
                         (b, n, self.num_heads, -1))
        v = self.transpose(v, (0, 2, 1, 3))

        attn = self.matmul(q, self.transpose(k, (0, 1, 3, 2))
                           ) * self.scale
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = self.transpose(self.matmul(attn, v), (0, 2, 1, 3))
        x = self.reshape(x, (b, n, -1))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Cell):
    """
    Multi-layer perceptron

    Args:
        in_features(int): Number of input features
        hidden_features(int): Number of hidden features
        out_features(int): Number of output features
        act_layer(class): Activation layer (base class)
        drop(float): Dropout rate
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Dense(in_channels=in_features, out_channels=hidden_features, has_bias=True)
        self.act = act_layer()
        self.fc2 = nn.Dense(in_channels=hidden_features, out_channels=out_features, has_bias=True)
        self.drop = nn.Dropout(keep_prob=1.0 - drop)  # if drop > 0. else Identity()

    def construct(self, *inputs, **kwargs):
        x = inputs[0]
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SE(nn.Cell):
    """SE Block"""

    def __init__(self, dim, hidden_ratio=None):
        super().__init__()
        hidden_ratio = hidden_ratio or 1
        self.dim = dim
        hidden_dim = int(dim * hidden_ratio)
        self.fc = nn.SequentialCell([
            nn.LayerNorm(normalized_shape=dim, epsilon=1e-5),
            nn.Dense(in_channels=dim, out_channels=hidden_dim),
            nn.ReLU(),
            nn.Dense(in_channels=hidden_dim, out_channels=dim),
            nn.Tanh()
        ])

        self.reduce_mean = P.ReduceMean()

    def construct(self, *inputs, **kwargs):
        x = inputs[0]
        a = self.reduce_mean(True, x, 1)  # B, 1, C
        a = self.fc(a)
        x = a * x
        return x


class Block(nn.Cell):
    """
    TNT base block

    Args:
        outer_dim(int): Number of output features
        inner_dim(int): Number of internal features
        outer_num_heads(int): Number of output heads
        inner_num_heads(int): Number of internal heads
        num_words(int): Number of 'visual words' (feature groups)
        mlp_ratio(float): Rate of MLP per hidden features
        qkv_bias(bool): Use Qk / v bias
        qk_scale(float): Qk scale
        drop(float): Dropout rate
        attn_drop(float): Dropout rate of attention layer
        drop_path(float): Path dropout rate
        act_layer(class): Activation layer (class)
        norm_layer(class): Normalization layer
        se(int): SE parameter
    """

    def __init__(self, outer_dim, inner_dim, outer_num_heads,
                 inner_num_heads, num_words, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, se=0):
        super().__init__()
        self.has_inner = inner_dim > 0
        if self.has_inner:
            # Inner
            self.inner_norm1 = norm_layer((inner_dim,), epsilon=1e-5)
            self.inner_attn = Attention(
                inner_dim, inner_dim, num_heads=inner_num_heads, qkv_bias=qkv_bias,
                qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
            self.inner_norm2 = norm_layer((inner_dim,), epsilon=1e-5)
            self.inner_mlp = Mlp(in_features=inner_dim, hidden_features=int(inner_dim * mlp_ratio),
                                 out_features=inner_dim, act_layer=act_layer, drop=drop)

            self.proj_norm1 = norm_layer((num_words * inner_dim,), epsilon=1e-5)
            self.proj = nn.Dense(in_channels=num_words * inner_dim, out_channels=outer_dim, has_bias=False)
            self.proj_norm2 = norm_layer((outer_dim,), epsilon=1e-5)
        # Outer
        self.outer_norm1 = norm_layer((outer_dim,), epsilon=1e-5)
        self.outer_attn = Attention(
            outer_dim, outer_dim, num_heads=outer_num_heads, qkv_bias=qkv_bias,
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath1D(drop_path)
        self.outer_norm2 = norm_layer((outer_dim,), epsilon=1e-5)
        self.outer_mlp = Mlp(in_features=outer_dim, hidden_features=int(outer_dim * mlp_ratio),
                             out_features=outer_dim, act_layer=act_layer, drop=drop)
        # SE
        self.se = se
        self.se_layer = 0
        if self.se > 0:
            self.se_layer = SE(outer_dim, 0.25)
        self.zeros = Tensor(np.zeros([1, 1, 1]), dtype=mstype.float32)

        self.reshape = P.Reshape()
        self.cast = P.Cast()

    def construct(self, *inputs, **kwargs):
        """TNT Block construct"""

        inner_tokens, outer_tokens = inputs[0], inputs[1]
        if self.has_inner:
            in1 = self.inner_norm1(inner_tokens)
            attn1 = self.inner_attn(in1)
            inner_tokens = inner_tokens + self.drop_path(attn1)  # B*N, k*k, c
            in2 = self.inner_norm2(inner_tokens)
            mlp = self.inner_mlp(in2)
            inner_tokens = inner_tokens + self.drop_path(mlp)  # B*N, k*k, c
            b, n, _ = P.Shape()(outer_tokens)
            # zeros = P.Tile()(self.zeros, (B, 1, C))
            proj = self.proj_norm2(self.proj(self.proj_norm1(
                self.reshape(inner_tokens, (b, n - 1, -1,))
            )))
            proj = self.cast(proj, mstype.float32)
            # proj = P.Concat(1)((zeros, proj))
            # outer_tokens = outer_tokens + proj  # B, N, C
            outer_tokens[:, 1:] = outer_tokens[:, 1:] + proj
        if self.se > 0:
            outer_tokens = outer_tokens + self.drop_path(
                self.outer_attn(self.outer_norm1(outer_tokens)))
            tmp_ = self.outer_mlp(self.outer_norm2(outer_tokens))
            outer_tokens = outer_tokens + self.drop_path(
                tmp_ + self.se_layer(tmp_))
        else:
            outer_tokens = outer_tokens + self.drop_path(
                self.outer_attn(self.outer_norm1(outer_tokens)))
            outer_tokens = outer_tokens + self.drop_path(
                self.outer_mlp(self.outer_norm2(outer_tokens)))
        return inner_tokens, outer_tokens


class TNT(nn.Cell):
    """
    TNT (Transformer in Transformer) for computer vision

    Args:
        img_size(int): Image size (side, px)
        patch_size(int): Patch size (side, px)
        in_chans(int): Number of input channels
        num_classes(int): Number of output classes
        outer_dim(int): Number of output features
        inner_dim(int): Number of internal features
        depth(int): Number of TNT base blocks
        outer_num_heads(int): Number of output heads
        inner_num_heads(int): Number of internal heads
        mlp_ratio(float): Rate of MLP per hidden features
        qkv_bias(bool): Use Qk / v bias
        qk_scale(float): Qk scale
        drop_rate(float): Dropout rate
        attn_drop_rate(float): Dropout rate for attention layer
        drop_path_rate(float): Dropout rate for DropPath layer
        norm_layer(class): Normalization layer
        inner_stride(int): Number of strides for internal patches
        se(int): SE parameter
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 num_classes=1000, outer_dim=768, inner_dim=48,
                 depth=12, outer_num_heads=12, inner_num_heads=4,
                 mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, inner_stride=4, se=0,
                 **kwargs):
        super().__init__()
        _ = kwargs
        self.num_classes = num_classes
        self.outer_dim = outer_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, outer_dim=outer_dim,
            inner_dim=inner_dim, inner_stride=inner_stride)
        self.num_patches = num_patches = self.patch_embed.num_patches
        num_words = self.patch_embed.num_words

        self.proj_norm1 = norm_layer((num_words * inner_dim,), epsilon=1e-5)
        self.proj = nn.Dense(in_channels=num_words * inner_dim, out_channels=outer_dim, has_bias=True)
        self.proj_norm2 = norm_layer((outer_dim,), epsilon=1e-5)

        self.cls_token = Parameter(Tensor(trunc_array([1, 1, outer_dim]), dtype=mstype.float32), name="cls_token",
                                   requires_grad=True)
        self.outer_pos = Parameter(Tensor(trunc_array([1, num_patches + 1, outer_dim]), dtype=mstype.float32),
                                   name="outer_pos")
        self.inner_pos = Parameter(Tensor(trunc_array([1, num_words, inner_dim]), dtype=mstype.float32))
        self.pos_drop = nn.Dropout(keep_prob=1.0 - drop_rate)

        dpr = [x for x in np.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        vanilla_idxs = []
        blocks = []
        for i in range(depth):
            if i in vanilla_idxs:
                blocks.append(Block(
                    outer_dim=outer_dim, inner_dim=-1, outer_num_heads=outer_num_heads, inner_num_heads=inner_num_heads,
                    num_words=num_words, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                    attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, se=se))
            else:
                blocks.append(Block(
                    outer_dim=outer_dim, inner_dim=inner_dim, outer_num_heads=outer_num_heads,
                    inner_num_heads=inner_num_heads,
                    num_words=num_words, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                    attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, se=se))
        self.blocks = nn.CellList(blocks)
        # self.norm = norm_layer(outer_dim, eps=1e-5)
        self.norm = norm_layer((outer_dim,))

        # NOTE as per official impl, we could have a pre-logits representation dense layer + tanh here
        # self.repr = nn.Linear(outer_dim, representation_size)
        # self.repr_act = nn.Tanh()

        # Classifier head
        mask = np.zeros([1, num_patches + 1, 1])
        mask[:, 0] = 1
        self.mask = Tensor(mask, dtype=mstype.float32)
        self.head = nn.Dense(in_channels=outer_dim, out_channels=num_classes, has_bias=True)

        self.reshape = P.Reshape()
        self.concat = P.Concat(1)
        self.tile = P.Tile()
        self.cast = P.Cast()

        self.init_weights()
        print("================================success================================")

    def init_weights(self):
        """init_weights"""
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(weight_init.initializer(weight_init.TruncatedNormal(sigma=0.02),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))
                if isinstance(cell, nn.Dense) and cell.bias is not None:
                    cell.bias.set_data(weight_init.initializer(weight_init.Zero(),
                                                               cell.bias.shape,
                                                               cell.bias.dtype))
            elif isinstance(cell, nn.LayerNorm):
                cell.gamma.set_data(weight_init.initializer(weight_init.One(),
                                                            cell.gamma.shape,
                                                            cell.gamma.dtype))
                cell.beta.set_data(weight_init.initializer(weight_init.Zero(),
                                                           cell.beta.shape,
                                                           cell.beta.dtype))

    def forward_features(self, x):
        """TNT forward_features"""
        b = x.shape[0]
        inner_tokens = self.patch_embed(x) + self.inner_pos  # B*N, 8*8, C

        outer_tokens = self.proj_norm2(
            self.proj(self.proj_norm1(
                self.reshape(inner_tokens, (b, self.num_patches, -1,))
            ))
        )
        outer_tokens = self.cast(outer_tokens, mstype.float32)
        outer_tokens = self.concat((
            self.tile(self.cls_token, (b, 1, 1)), outer_tokens
        ))

        outer_tokens = outer_tokens + self.outer_pos
        outer_tokens = self.pos_drop(outer_tokens)

        for blk in self.blocks:
            inner_tokens, outer_tokens = blk(inner_tokens, outer_tokens)

        outer_tokens = self.norm(outer_tokens)  # [batch_size, num_patch+1, outer_dim)
        return outer_tokens[:, 0]

    def construct(self, *inputs, **kwargs):
        x = inputs[0]
        x = self.forward_features(x)
        x = self.head(x)
        return x


@register_model
def tnt_small(pretrained: bool = False, num_classes=1000, in_channels=3, **kwargs):
    """tnt_s_patch16_224"""

    patch_size = 16
    inner_stride = 4
    outer_dim = 384
    inner_dim = 24
    outer_num_heads = 6
    inner_num_heads = 4
    depth = 12
    num_classes = num_classes
    outer_dim = make_divisible(outer_dim, outer_num_heads)
    inner_dim = make_divisible(inner_dim, inner_num_heads)
    model = TNT(patch_size=patch_size, in_chans=in_channels, num_classes=num_classes,
                outer_dim=outer_dim, inner_dim=inner_dim, depth=depth,
                outer_num_heads=outer_num_heads, inner_num_heads=inner_num_heads, qkv_bias=False,
                inner_stride=inner_stride, **kwargs)
    default_cfg = default_cfgs["tnt_small"]

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def tnt_base(pretrained: bool = False, num_classes=1000, in_channels=3, **kwargs):
    """tnt_b_patch16_224"""

    patch_size = 16
    inner_stride = 4
    outer_dim = 640
    inner_dim = 40
    outer_num_heads = 10
    inner_num_heads = 4
    depth = 12
    num_classes = num_classes
    outer_dim = make_divisible(outer_dim, outer_num_heads)
    inner_dim = make_divisible(inner_dim, inner_num_heads)
    model = TNT(patch_size=patch_size, in_chans=in_channels, num_classes=num_classes,
                outer_dim=outer_dim, inner_dim=inner_dim, depth=depth,
                outer_num_heads=outer_num_heads, inner_num_heads=inner_num_heads, qkv_bias=False,
                inner_stride=inner_stride, **kwargs)
    default_cfg = default_cfgs["tnt_base"]

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model
