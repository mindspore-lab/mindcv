"""ViT"""
import math
from typing import List, Optional, Union

import numpy as np

from mindspore import Parameter, nn, ops
from mindspore.common.initializer import TruncatedNormal, initializer

from .helpers import ConfigDict, load_pretrained
from .layers.attention import Attention
from .layers.compatibility import Dropout
from .layers.drop_path import DropPath
from .layers.mlp import Mlp
from .layers.patch_embed import PatchEmbed
from .layers.pos_embed import RelativePositionBiasWithCLS
from .registry import register_model

__all__ = [
    "VisionTransformerEncoder",
    "ViT",
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
        "classifier": "head.classifier",
        **kwargs,
    }


default_cfgs = {
    "vit_b_16_224": _cfg(url=""),
    "vit_b_16_384": _cfg(
        url="", input_size=(3, 384, 384)
    ),
    "vit_l_16_224": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/vit/vit_l_16_224-97d0fdbc.ckpt"),
    "vit_l_16_384": _cfg(
        url="", input_size=(3, 384, 384)
    ),
    "vit_b_32_224": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/vit/vit_b_32_224-f50866e8.ckpt"),
    "vit_b_32_384": _cfg(
        url="", input_size=(3, 384, 384)
    ),
    "vit_l_32_224": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/vit/vit_l_32_224-b80441df.ckpt"),
}


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


class TransformerBlock(nn.Cell):
    """
    Transformer block implementation.

    Args:
        dim (int): The dimension of embedding.
        num_heads (int): The number of attention heads.
        qkv_bias (bool): Specifies whether the linear layer uses a bias vector. Default: True.
        qk_scale: (float): The user-defined factor to scale the product of q and k. Default: None.
        attn_drop (float): The drop rate of attention, greater than 0 and less equal than 1. Default: 0.0.
        proj_drop (float): The drop rate of dense layer output, greater than 0 and less equal than 1. Default: 0.0.
        attn_head_dim (int): The user-defined dimension of attention head features. Default: None.
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
        super(TransformerBlock, self).__init__()
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
    '''
    ViT encoder, which returns the feature encoded by transformer encoder.
    '''
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
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
        use_rel_pos_emb: bool = False,
        use_rel_pos_bias: bool = False,
        use_shared_rel_pos_bias: bool = True,
        **kwargs
    ):
        super(VisionTransformerEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(image_size=image_size, patch_size=patch_size,
                                      in_chans=in_channels, embed_dim=embed_dim)
        self.num_patches = self.patch_embed.num_patches

        self.cls_token = Parameter(initializer(TruncatedNormal(0.02), (1, 1, embed_dim)))

        self.pos_embed = Parameter(initializer(TruncatedNormal(0.02),
                                   (1, self.num_patches + 1, embed_dim))) if not use_rel_pos_emb else None
        self.pos_drop = Dropout(pos_drop_rate)

        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBiasWithCLS(
                    window_size=self.patch_embed.patches_resolution,
                    num_heads=num_heads,
                    )
        elif use_rel_pos_bias:
            self.rel_pos_bias = nn.CellList([
                RelativePositionBiasWithCLS(window_size=self.patch_embed.patches_resolution,
                                            num_heads=num_heads) for _ in range(depth)
            ])
        else:
            self.rel_pos_bias = None

        dpr = [x.item() for x in np.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.CellList([
            TransformerBlock(
                dim=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop_rate, proj_drop=proj_drop_rate, attn_head_dim=attn_head_dim,
                mlp_ratio=mlp_ratio, drop_path=dpr[i], init_values=init_values,
                act_layer=act_layer, norm_layer=norm_layer
            ) for i in range(depth)
        ])

        self._init_weights()
        self._fix_init_weights()

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

        cls_tokens = ops.broadcast_to(self.cls_token, (bsz, -1, -1))
        cls_tokens = cls_tokens.astype(x.dtype)
        x = ops.concat((cls_tokens, x), axis=1)

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
        x = self.forward_features(x)
        return x


class DenseHead(nn.Cell):
    """
    LinearClsHead architecture.

    Args:
        input_channel (int): The number of input channel.
        num_classes (int): Number of classes.
        has_bias (bool): Specifies whether the layer uses a bias vector. Default: True.
        activation (Union[str, Cell, Primitive]): activate function applied to the output. Eg. `ReLU`. Default: None.
        keep_prob (float): Dropout keeping rate, between [0, 1]. E.g. rate=0.9, means dropping out 10% of input.
            Default: 1.0.

    Returns:
        Tensor, output tensor.
    """

    def __init__(
        self,
        input_channel: int,
        num_classes: int,
        has_bias: bool = True,
        activation: Optional[Union[str, nn.Cell]] = None,
        keep_prob: float = 1.0,
    ) -> None:
        super().__init__()

        self.dropout = Dropout(p=1.0-keep_prob)
        self.classifier = nn.Dense(input_channel, num_classes, has_bias=has_bias, activation=activation)

    def construct(self, x):
        if self.training:
            x = self.dropout(x)
        x = self.classifier(x)
        return x


class MultilayerDenseHead(nn.Cell):
    """
    MultilayerDenseHead architecture.

    Args:
        input_channel (int): The number of input channel.
        num_classes (int): Number of classes.
        mid_channel (list): Number of channels in the hidden fc layers.
        keep_prob (list): Dropout keeping rate, between [0, 1]. E.g. rate=0.9, means dropping out 10% of
        input.
        activation (list): activate function applied to the output. Eg. `ReLU`.

    Returns:
        Tensor, output tensor.
    """

    def __init__(
        self,
        input_channel: int,
        num_classes: int,
        mid_channel: List[int],
        keep_prob: List[float],
        activation: List[Optional[Union[str, nn.Cell]]],
    ) -> None:
        super().__init__()
        mid_channel.append(num_classes)
        assert len(mid_channel) == len(activation) == len(keep_prob), "The length of the list should be the same."

        length = len(activation)
        head = []

        for i in range(length):
            linear = DenseHead(
                input_channel,
                mid_channel[i],
                activation=activation[i],
                keep_prob=keep_prob[i],
            )
            head.append(linear)
            input_channel = mid_channel[i]

        self.classifier = nn.SequentialCell(head)

    def construct(self, x):
        x = self.classifier(x)

        return x


class ViT(VisionTransformerEncoder):
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
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
        use_rel_pos_emb: bool = False,
        use_rel_pos_bias: bool = False,
        use_shared_rel_pos_bias: bool = True,
        use_cls: bool = True,
        representation_size: Optional[int] = None,
        num_classes: int = 1000,
        **kwargs
    ):
        super(ViT, self).__init__(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            attn_head_dim=attn_head_dim,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            pos_drop_rate=pos_drop_rate,
            proj_drop_rate=proj_drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            init_values=init_values,
            act_layer=act_layer,
            norm_layer=norm_layer,
            use_rel_pos_emb=use_rel_pos_emb,
            use_rel_pos_bias=use_rel_pos_bias,
            use_shared_rel_pos_bias=use_shared_rel_pos_bias,
            **kwargs
        )
        self.use_cls = use_cls
        self.norm = norm_layer((embed_dim,))

        if representation_size:
            self.head = MultilayerDenseHead(
                input_channel=embed_dim,
                num_classes=num_classes,
                mid_channel=[representation_size],
                activation=["tanh", None],
                keep_prob=[1.0, 1.0],
            )
        else:
            self.head = DenseHead(input_channel=embed_dim, num_classes=num_classes)

    def construct(self, x):
        x = self.forward_features(x)
        x = self.norm(x)

        if self.use_cls:
            x = x[:, 0]
        else:
            x = x[:, 1:].mean(axis=1)

        x = self.head(x)
        return x


def vit(
    image_size: int = 224,
    patch_size: int = 16,
    in_channels: int = 3,
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
    init_values: Optional[float] = None,
    act_layer: nn.Cell = nn.GELU,
    norm_layer: nn.Cell = nn.LayerNorm,
    use_rel_pos_emb: bool = False,
    use_rel_pos_bias: bool = False,
    use_shared_rel_pos_bias: bool = False,
    use_cls: bool = True,
    representation_size: Optional[int] = None,
    num_classes: int = 1000,
    pretrained: bool = False,
    url_cfg: dict = None,
) -> ViT:

    """Vision Transformer architecture."""

    model = ViT(
        image_size=image_size,
        patch_size=patch_size,
        in_channels=in_channels,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        attn_head_dim=attn_head_dim,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        qk_scale=qk_scale,
        pos_drop_rate=pos_drop_rate,
        proj_drop_rate=proj_drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        init_values=init_values,
        act_layer=act_layer,
        norm_layer=norm_layer,
        use_rel_pos_emb=use_rel_pos_emb,
        use_rel_pos_bias=use_rel_pos_bias,
        use_shared_rel_pos_bias=use_shared_rel_pos_bias,
        use_cls=use_cls,
        representation_size=representation_size,
        num_classes=num_classes
    )

    if pretrained:
        # Download the pre-trained checkpoint file from url, and load ckpt file.
        load_pretrained(model, url_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def vit_b_16_224(
    pretrained: bool = False,
    num_classes: int = 1000,
    in_channels: int = 3,
    has_logits: bool = False,
    drop_rate: float = 0.0,
    drop_path_rate: float = 0.0,
):
    config = ConfigDict()
    config.image_size = 224
    config.patch_size = 16
    config.in_channels = in_channels
    config.embed_dim = 768
    config.depth = 12
    config.num_heads = 12
    config.pos_drop_rate = drop_rate
    config.proj_drop_rate = drop_rate
    config.attn_drop_rate = drop_rate
    config.drop_path_rate = drop_path_rate
    config.representation_size = 768 if has_logits else None
    config.num_classes = num_classes

    config.pretrained = pretrained
    config.url_cfg = default_cfgs["vit_b_16_224"]

    return vit(**config)


@register_model
def vit_b_16_384(
    pretrained: bool = False,
    num_classes: int = 1000,
    in_channels: int = 3,
    has_logits: bool = False,
    drop_rate: float = 0.0,
    drop_path_rate: float = 0.0,
):
    config = ConfigDict()
    config.image_size = 384
    config.patch_size = 16
    config.in_channels = in_channels
    config.embed_dim = 768
    config.depth = 12
    config.num_heads = 12
    config.pos_drop_rate = drop_rate
    config.proj_drop_rate = drop_rate
    config.attn_drop_rate = drop_rate
    config.drop_path_rate = drop_path_rate
    config.representation_size = 768 if has_logits else None
    config.num_classes = num_classes

    config.pretrained = pretrained
    config.url_cfg = default_cfgs["vit_b_16_384"]

    return vit(**config)


@register_model
def vit_l_16_224(
    pretrained: bool = False,
    num_classes: int = 1000,
    in_channels: int = 3,
    has_logits: bool = False,
    drop_rate: float = 0.0,
    drop_path_rate: float = 0.0,
):
    config = ConfigDict()
    config.image_size = 224
    config.patch_size = 16
    config.in_channels = in_channels
    config.embed_dim = 1024
    config.depth = 24
    config.num_heads = 16
    config.pos_drop_rate = drop_rate
    config.proj_drop_rate = drop_rate
    config.attn_drop_rate = drop_rate
    config.drop_path_rate = drop_path_rate
    config.representation_size = 1024 if has_logits else None
    config.num_classes = num_classes

    config.pretrained = pretrained
    config.url_cfg = default_cfgs["vit_l_16_224"]

    return vit(**config)


@register_model
def vit_l_16_384(
    pretrained: bool = False,
    num_classes: int = 1000,
    in_channels: int = 3,
    has_logits: bool = False,
    drop_rate: float = 0.0,
    drop_path_rate: float = 0.0,
):
    config = ConfigDict()
    config.image_size = 384
    config.patch_size = 16
    config.in_channels = in_channels
    config.embed_dim = 1024
    config.depth = 24
    config.num_heads = 16
    config.pos_drop_rate = drop_rate
    config.proj_drop_rate = drop_rate
    config.attn_drop_rate = drop_rate
    config.drop_path_rate = drop_path_rate
    config.representation_size = 1024 if has_logits else None
    config.num_classes = num_classes

    config.pretrained = pretrained
    config.url_cfg = default_cfgs["vit_l_16_384"]

    return vit(**config)


@register_model
def vit_b_32_224(
    pretrained: bool = False,
    num_classes: int = 1000,
    in_channels: int = 3,
    has_logits: bool = False,
    drop_rate: float = 0.0,
    drop_path_rate: float = 0.0,
):
    config = ConfigDict()
    config.image_size = 224
    config.patch_size = 32
    config.in_channels = in_channels
    config.embed_dim = 768
    config.depth = 12
    config.num_heads = 12
    config.pos_drop_rate = drop_rate
    config.proj_drop_rate = drop_rate
    config.attn_drop_rate = drop_rate
    config.drop_path_rate = drop_path_rate
    config.representation_size = 768 if has_logits else None
    config.num_classes = num_classes

    config.pretrained = pretrained
    config.url_cfg = default_cfgs["vit_b_32_224"]

    return vit(**config)


@register_model
def vit_b_32_384(
    pretrained: bool = False,
    num_classes: int = 1000,
    in_channels: int = 3,
    has_logits: bool = False,
    drop_rate: float = 0.0,
    drop_path_rate: float = 0.0,
):
    config = ConfigDict()
    config.image_size = 384
    config.patch_size = 32
    config.in_channels = in_channels
    config.embed_dim = 768
    config.depth = 12
    config.num_heads = 12
    config.pos_drop_rate = drop_rate
    config.proj_drop_rate = drop_rate
    config.attn_drop_rate = drop_rate
    config.drop_path_rate = drop_path_rate
    config.representation_size = 768 if has_logits else None
    config.num_classes = num_classes

    config.pretrained = pretrained
    config.url_cfg = default_cfgs["vit_b_32_384"]

    return vit(**config)


@register_model
def vit_l_32_224(
    pretrained: bool = False,
    num_classes: int = 1000,
    in_channels: int = 3,
    has_logits: bool = False,
    drop_rate: float = 0.0,
    drop_path_rate: float = 0.0,
):
    config = ConfigDict()
    config.image_size = 224
    config.patch_size = 32
    config.in_channels = in_channels
    config.embed_dim = 1024
    config.depth = 24
    config.num_heads = 16
    config.pos_drop_rate = drop_rate
    config.proj_drop_rate = drop_rate
    config.attn_drop_rate = drop_rate
    config.drop_path_rate = drop_path_rate
    config.representation_size = 1024 if has_logits else None
    config.num_classes = num_classes

    config.pretrained = pretrained
    config.url_cfg = default_cfgs["vit_l_32_224"]

    return vit(**config)
