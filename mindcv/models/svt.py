"""SVT model implementation"""

from functools import partial

import numpy as np

import mindspore as ms
import mindspore.ops.functional as F
from mindspore import Parameter, nn

from .layers import DropPath, Identity, Mlp
from .pcpvt import CPVTV2, Attention, GroupAttention, _cfg
from .registry import register_model
from .utils import load_pretrained

__all__ = [
    "svt_small",
    "svt_base",
    "svt_large"
]


default_cfgs = {
    "svt_small": _cfg(url="https://storage.googleapis.com/huawei-mindspore-hk/Twins/converted/svt_s_new.ckpt"),
    "svt_base": _cfg(url="https://storage.googleapis.com/huawei-mindspore-hk/Twins/converted/svt_b_new.ckpt"),
    "svt_large": _cfg(url="https://storage.googleapis.com/huawei-mindspore-hk/Twins/converted/svt_l_new.ckpt")
}


class LayerScale(nn.Cell):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = Parameter(init_values * F.ones(dim, ms.dtype.float32))

    def construct(self, *inputs, **kwargs):
        x = inputs[0]
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Cell):
    """Base block class for SVT model"""

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, name=''):
        super().__init__()
        self.norm1 = norm_layer([dim])
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            name=name + '_attn'
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        self.norm2 = norm_layer([dim])
        self.norm2.beta.name = '{}_attn.{}'.format(name, self.norm2.beta)
        self.norm2.gamma.name = '{}_attn.{}'.format(name, self.norm2.gamma)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def construct(self, *inputs, **kwargs):
        x = inputs[0]
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SBlock(Block):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super(SBlock, self).__init__(dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop,
                                     drop_path, act_layer, norm_layer)
        _ = sr_ratio  # for compatibility with other blocks

    def construct(self, *inputs, **kwargs):
        x = inputs[0]
        return super(SBlock, self).forward(x)


class GroupBlock(Block):
    """Implementation of group-aggregation block for SVT"""

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, ws=1,
                 name='', h=0, w=0):
        super(GroupBlock, self).__init__(dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop,
                                         drop_path, act_layer, norm_layer)
        del self.attn
        if ws == 1:
            self.attn = Attention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, sr_ratio, name=name, h=h, w=w)
        else:
            self.attn = GroupAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, ws, name=name, h=h, w=w)
        self.h, self.w = h, w

    def construct(self, *inputs, **kwargs):
        x = inputs[0]
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PCPVT(CPVTV2):
    """PCPVT wrapper with some default arguments"""

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dims=(64, 128, 256),
                 num_heads=(1, 2, 4), mlp_ratios=(4, 4, 4), qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=(4, 4, 4), sr_ratios=(4, 2, 1), block_cls=SBlock):
        super(PCPVT, self).__init__(img_size, patch_size, in_chans, num_classes, embed_dims, num_heads,
                                    mlp_ratios, qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate,
                                    norm_layer, depths, sr_ratios, block_cls)


class ALTGVT(PCPVT):
    """Twins SVT model"""
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dims=(64, 128, 256),
                 num_heads=(1, 2, 4), mlp_ratios=(4, 4, 4), qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=(4, 4, 4), sr_ratios=(4, 2, 1), block_cls=GroupBlock, wss=(7, 7, 7)):
        super(ALTGVT, self).__init__(img_size, patch_size, in_chans, num_classes, embed_dims, num_heads,
                                     mlp_ratios, qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate,
                                     norm_layer, depths, sr_ratios, block_cls)
        del self.blocks
        self.wss = wss
        # transformer encoder
        dpr = np.linspace(0, drop_path_rate, sum(depths))  # stochastic depth decay rule
        cur = 0
        self.blocks = []
        for k in range(len(depths)):
            block = nn.CellList([block_cls(
                dim=embed_dims[k], num_heads=num_heads[k], mlp_ratio=mlp_ratios[k], qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
                sr_ratio=sr_ratios[k], ws=1 if i % 2 == 1 else wss[k],
                name='b{}.{}'.format(k, i), h=self.patch_embeds[k].h, w=self.patch_embeds[k].w
            ) for i in range(depths[k])])
            self.blocks.extend(block)
            cur += depths[k]

        self.merge_blocks = nn.CellList()
        total = 0
        self.inds = []
        for k, d in enumerate(self.depths):
            self.merge_blocks.append(nn.SequentialCell([
                self.blocks[total],
                self.pos_block[k]
            ] + self.blocks[total + 1:total + d]))
            self.inds.append([total, -1 - k] + list(range(total + 1, total + d)))
            total += d
        # self.apply(self._init_weights)


@register_model
def svt_small(pretrained: bool = False, num_classes=1000, in_channels=3, **kwargs) -> ALTGVT:
    model = ALTGVT(
        patch_size=4, in_chans=in_channels, num_classes=num_classes,
        embed_dims=(64, 128, 256, 512), num_heads=(2, 4, 8, 16), mlp_ratios=(4, 4, 4, 4), qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), depths=(2, 2, 10, 4), wss=(7, 7, 7, 7), sr_ratios=(8, 4, 2, 1),
        **kwargs)
    default_cfg = default_cfgs["svt_small"]

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def svt_base(pretrained: bool = False, num_classes=1000, in_channels=3, **kwargs) -> ALTGVT:
    model = ALTGVT(
        patch_size=4, in_chans=in_channels, num_classes=num_classes,
        embed_dims=(96, 192, 384, 768), num_heads=(3, 6, 12, 24), mlp_ratios=(4, 4, 4, 4), qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), depths=(2, 2, 18, 2), wss=(7, 7, 7, 7), sr_ratios=(8, 4, 2, 1),
        **kwargs)
    default_cfg = default_cfgs["svt_base"]

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def svt_large(pretrained: bool = False, num_classes=1000, in_channels=3, **kwargs) -> ALTGVT:
    model = ALTGVT(
        patch_size=4, in_chans=in_channels, num_classes=num_classes,
        embed_dims=(128, 256, 512, 1024), num_heads=(4, 8, 16, 32), mlp_ratios=(4, 4, 4, 4), qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), depths=(2, 2, 18, 2), wss=(7, 7, 7, 7), sr_ratios=(8, 4, 2, 1),
        **kwargs)
    default_cfg = default_cfgs["svt_large"]

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model
