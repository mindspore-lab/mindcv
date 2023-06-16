"""
MindSpore implementation of `ConvNeXt` and `ConvNeXt V2`.
Refer to: A ConvNet for the 2020s
          ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders
"""
from typing import List, Tuple

import numpy as np

import mindspore.common.initializer as init
from mindspore import Parameter, Tensor
from mindspore import dtype as mstype
from mindspore import nn, ops

from .helpers import build_model_with_cfg
from .layers.drop_path import DropPath
from .layers.identity import Identity
from .registry import register_model

__all__ = [
    "ConvNeXt",
    "convnext_tiny",
    "convnext_small",
    "convnext_base",
    "convnext_large",
    "convnext_xlarge",
    "convnextv2_atto",
    "convnextv2_femto",
    "convnextv2_pico",
    "convnextv2_nano",
    "convnextv2_tiny",
    "convnextv2_base",
    "convnextv2_large",
    "convnextv2_huge",
]


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "first_conv": "feature.0.0",
        "classifier": "classifier",
        **kwargs,
    }


default_cfgs = {
    "convnext_tiny": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/convnext/convnext_tiny-ae5ff8d7.ckpt"),
    "convnext_small": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/convnext/convnext_small-e23008f3.ckpt"),
    "convnext_base": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/convnext/convnext_base-ee3544b8.ckpt"),
    "convnext_large": _cfg(url=""),
    "convnext_xlarge": _cfg(url=""),
    "convnextv2_atto": _cfg(url=""),
    "convnextv2_femto": _cfg(url=""),
    "convnextv2_pico": _cfg(url=""),
    "convnextv2_nano": _cfg(url=""),
    "convnextv2_tiny": _cfg(url=""),
    "convnextv2_base": _cfg(url=""),
    "convnextv2_large": _cfg(url=""),
    "convnextv2_huge": _cfg(url=""),
}


class GRN(nn.Cell):
    """ GRN (Global Response Normalization) layer """

    def __init__(self, dim: int):
        super().__init__()
        self.gamma = Parameter(Tensor(np.zeros([1, 1, 1, dim]), mstype.float32))
        self.beta = Parameter(Tensor(np.zeros([1, 1, 1, dim]), mstype.float32))
        self.norm = ops.LpNorm(axis=[1, 2], p=2, keep_dims=True)

    def construct(self, x: Tensor) -> Tensor:
        gx = self.norm(x)
        nx = gx / (ops.mean(gx, axis=-1, keep_dims=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x


class ConvNextLayerNorm(nn.LayerNorm):
    """
    LayerNorm for channels_first tensors with 2d spatial dimensions (ie N, C, H, W).
    """

    def __init__(
        self,
        normalized_shape: Tuple[int],
        epsilon: float,
        norm_axis: int = -1,
    ) -> None:
        super().__init__(normalized_shape=normalized_shape, epsilon=epsilon)
        assert norm_axis in (-1, 1), "ConvNextLayerNorm's norm_axis must be 1 or -1."
        self.norm_axis = norm_axis

    def construct(self, input_x: Tensor) -> Tensor:
        if self.norm_axis == -1:
            y, _, _ = self.layer_norm(input_x, self.gamma, self.beta)
        else:
            input_x = ops.transpose(input_x, (0, 2, 3, 1))
            y, _, _ = self.layer_norm(input_x, self.gamma, self.beta)
            y = ops.transpose(y, (0, 3, 1, 2))
        return y


class Block(nn.Cell):
    """ConvNeXt Block
    There are two equivalent implementations:
      (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
      (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    Unlike the official impl, this one allows choice of 1 or 2, 1x1 conv can be faster with appropriate
    choice of LayerNorm impl, however as model size increases the tradeoffs appear to change and nn.Linear
    is a better choice. This was observed with PyTorch 1.10 on 3090 GPU, it could change over time & w/ different HW.
    Args:
        dim: Number of input channels.
        drop_path: Stochastic depth rate. Default: 0.0.
        layer_scale_init_value: Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        use_grn: bool = False,
    ) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, group=dim, has_bias=True)  # depthwise conv
        self.norm = ConvNextLayerNorm((dim,), epsilon=1e-6)
        self.pwconv1 = nn.Dense(dim, 4 * dim)  # pointwise/1x1 convs, implemented with Dense layers
        self.act = nn.GELU()
        self.use_grn = use_grn
        if use_grn:
            self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Dense(4 * dim, dim)
        self.gamma_ = Parameter(Tensor(layer_scale_init_value * np.ones((dim)), dtype=mstype.float32),
                                requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else Identity()

    def construct(self, x: Tensor) -> Tensor:
        downsample = x
        x = self.dwconv(x)
        x = ops.transpose(x, (0, 2, 3, 1))
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        if self.use_grn:
            x = self.grn(x)
        x = self.pwconv2(x)
        if self.gamma_ is not None:
            x = self.gamma_ * x
        x = ops.transpose(x, (0, 3, 1, 2))
        x = downsample + self.drop_path(x)
        return x


class ConvNeXt(nn.Cell):
    r"""ConvNeXt and ConvNeXt V2 model class, based on
    `"A ConvNet for the 2020s" <https://arxiv.org/abs/2201.03545>`_ and
    `"ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders" <https://arxiv.org/abs/2301.00808>`_

    Args:
        in_channels: dim of the input channel.
        num_classes: dim of the classes predicted.
        depths: the depths of each layer.
        dims: the middle dim of each layer.
        drop_path_rate: the rate of droppath. Default: 0.0.
        layer_scale_init_value: the parameter of init for the classifier. Default: 1e-6.
        head_init_scale: the parameter of init for the head. Default: 1.0.
        use_grn: If True, use Global Response Normalization in each block. Default: False.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        depths: List[int],
        dims: List[int],
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        head_init_scale: float = 1.0,
        use_grn: bool = False,
    ):
        super().__init__()

        downsample_layers = []  # stem and 3 intermediate down_sampling conv layers
        stem = nn.SequentialCell(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4, has_bias=True),
            ConvNextLayerNorm((dims[0],), epsilon=1e-6, norm_axis=1),
        )
        downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.SequentialCell(
                ConvNextLayerNorm((dims[i],), epsilon=1e-6, norm_axis=1),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2, has_bias=True),
            )
            downsample_layers.append(downsample_layer)

        total_reduction = 4
        self.feature_info = []
        self.flatten_sequential = True

        stages = []  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = list(np.linspace(0, drop_path_rate, sum(depths)))
        cur = 0
        for i in range(4):
            blocks = []
            for j in range(depths[i]):
                blocks.append(Block(dim=dims[i], drop_path=dp_rates[cur + j],
                                    layer_scale_init_value=layer_scale_init_value, use_grn=use_grn))
            stage = nn.SequentialCell(blocks)
            stages.append(stage)
            cur += depths[i]

            if i > 0:
                total_reduction *= 2
            self.feature_info.append(dict(chs=dims[i], reduction=total_reduction, name=f'feature.{i * 2 + 1}'))

        self.feature = nn.SequentialCell([
            downsample_layers[0],
            stages[0],
            downsample_layers[1],
            stages[1],
            downsample_layers[2],
            stages[2],
            downsample_layers[3],
            stages[3]
        ])
        self.norm = ConvNextLayerNorm((dims[-1],), epsilon=1e-6)  # final norm layer
        self.classifier = nn.Dense(dims[-1], num_classes)  # classifier
        self.head_init_scale = head_init_scale
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize weights for cells."""
        for _, cell in self.cells_and_names():
            if isinstance(cell, (nn.Dense, nn.Conv2d)):
                cell.weight.set_data(
                    init.initializer(init.TruncatedNormal(sigma=0.02), cell.weight.shape, cell.weight.dtype)
                )
                if isinstance(cell, nn.Dense) and cell.bias is not None:
                    cell.bias.set_data(init.initializer(init.Zero(), cell.bias.shape, cell.bias.dtype))
        self.classifier.weight.set_data(self.classifier.weight * self.head_init_scale)
        self.classifier.bias.set_data(self.classifier.bias * self.head_init_scale)

    def forward_head(self, x: Tensor) -> Tensor:
        x = self.classifier(x)
        return x

    def forward_features(self, x: Tensor) -> Tensor:
        x = self.feature(x)
        return self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)

    def construct(self, x: Tensor) -> Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _create_convnext(pretrained=False, **kwargs):
    return build_model_with_cfg(ConvNeXt, pretrained, **kwargs)


@register_model
def convnext_tiny(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> ConvNeXt:
    """Get ConvNeXt tiny model.
    Refer to the base class 'models.ConvNeXt' for more details.
    """
    default_cfg = default_cfgs["convnext_tiny"]
    model_args = dict(
        in_channels=in_channels, num_classes=num_classes, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs
    )
    return _create_convnext(pretrained, **dict(default_cfg=default_cfg, **model_args))


@register_model
def convnext_small(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> ConvNeXt:
    """Get ConvNeXt small model.
    Refer to the base class 'models.ConvNeXt' for more details.
    """
    default_cfg = default_cfgs["convnext_small"]
    model_args = dict(
        in_channels=in_channels, num_classes=num_classes, depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs
    )
    return _create_convnext(pretrained, **dict(default_cfg=default_cfg, **model_args))


@register_model
def convnext_base(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> ConvNeXt:
    """Get ConvNeXt base model.
    Refer to the base class 'models.ConvNeXt' for more details.
    """
    default_cfg = default_cfgs["convnext_base"]
    model_args = dict(
        in_channels=in_channels, num_classes=num_classes, depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs
    )
    return _create_convnext(pretrained, **dict(default_cfg=default_cfg, **model_args))


@register_model
def convnext_large(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> ConvNeXt:
    """Get ConvNeXt large model.
    Refer to the base class 'models.ConvNeXt' for more details.
    """
    default_cfg = default_cfgs["convnext_large"]
    model_args = dict(
        in_channels=in_channels, num_classes=num_classes, depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs
    )
    return _create_convnext(pretrained, **dict(default_cfg=default_cfg, **model_args))


@register_model
def convnext_xlarge(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> ConvNeXt:
    """Get ConvNeXt xlarge model.
    Refer to the base class 'models.ConvNeXt' for more details.
    """
    default_cfg = default_cfgs["convnext_xlarge"]
    model_args = dict(
        in_channels=in_channels, num_classes=num_classes, depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs
    )
    return _create_convnext(pretrained, **dict(default_cfg=default_cfg, **model_args))


@register_model
def convnextv2_atto(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> ConvNeXt:
    """Get ConvNeXt_v2 atto model.
    Refer to the base class 'models.ConvNeXt' for more details.
    """
    default_cfg = default_cfgs["convnextv2_atto"]
    model_args = dict(in_channels=in_channels, num_classes=num_classes, depths=[2, 2, 6, 2],
                      dims=[40, 80, 160, 320], use_grn=True, layer_scale_init_value=0.0, **kwargs)
    return _create_convnext(pretrained, **dict(default_cfg=default_cfg, **model_args))


@register_model
def convnextv2_femto(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> ConvNeXt:
    """Get ConvNeXt_v2 femto model.
    Refer to the base class 'models.ConvNeXt' for more details.
    """
    default_cfg = default_cfgs["convnextv2_femto"]
    model_args = dict(in_channels=in_channels, num_classes=num_classes, depths=[2, 2, 6, 2],
                      dims=[48, 96, 192, 384], use_grn=True, layer_scale_init_value=0.0, **kwargs)
    return _create_convnext(pretrained, **dict(default_cfg=default_cfg, **model_args))


@register_model
def convnextv2_pico(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> ConvNeXt:
    """Get ConvNeXt_v2 pico model.
    Refer to the base class 'models.ConvNeXt' for more details.
    """
    default_cfg = default_cfgs["convnextv2_pico"]
    model_args = dict(in_channels=in_channels, num_classes=num_classes, depths=[2, 2, 6, 2],
                      dims=[64, 128, 256, 512], use_grn=True, layer_scale_init_value=0.0, **kwargs)
    return _create_convnext(pretrained, **dict(default_cfg=default_cfg, **model_args))


@register_model
def convnextv2_nano(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> ConvNeXt:
    """Get ConvNeXt_v2 nano model.
    Refer to the base class 'models.ConvNeXt' for more details.
    """
    default_cfg = default_cfgs["convnextv2_nano"]
    model_args = dict(in_channels=in_channels, num_classes=num_classes, depths=[2, 2, 8, 2],
                      dims=[80, 160, 320, 640], use_grn=True, layer_scale_init_value=0.0, **kwargs)
    return _create_convnext(pretrained, **dict(default_cfg=default_cfg, **model_args))


@register_model
def convnextv2_tiny(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> ConvNeXt:
    """Get ConvNeXt_v2 tiny model.
    Refer to the base class 'models.ConvNeXt' for more details.
    """
    default_cfg = default_cfgs["convnextv2_tiny"]
    model_args = dict(in_channels=in_channels, num_classes=num_classes, depths=[3, 3, 9, 3],
                      dims=[96, 192, 384, 768], use_grn=True, layer_scale_init_value=0.0, **kwargs)
    return _create_convnext(pretrained, **dict(default_cfg=default_cfg, **model_args))


@register_model
def convnextv2_base(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> ConvNeXt:
    """Get ConvNeXt_v2 base model.
    Refer to the base class 'models.ConvNeXt' for more details.
    """
    default_cfg = default_cfgs["convnextv2_base"]
    model_args = dict(in_channels=in_channels, num_classes=num_classes, depths=[3, 3, 27, 3],
                      dims=[128, 256, 512, 1024], use_grn=True, layer_scale_init_value=0.0, **kwargs)
    return _create_convnext(pretrained, **dict(default_cfg=default_cfg, **model_args))


@register_model
def convnextv2_large(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> ConvNeXt:
    """Get ConvNeXt_v2 large model.
    Refer to the base class 'models.ConvNeXt' for more details.
    """
    default_cfg = default_cfgs["convnextv2_large"]
    model_args = dict(in_channels=in_channels, num_classes=num_classes, depths=[3, 3, 27, 3],
                      dims=[192, 384, 768, 1536], use_grn=True, layer_scale_init_value=0.0, **kwargs)
    return _create_convnext(pretrained, **dict(default_cfg=default_cfg, **model_args))


@register_model
def convnextv2_huge(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> ConvNeXt:
    """Get ConvNeXt_v2 huge model.
    Refer to the base class 'models.ConvNeXt' for more details.
    """
    default_cfg = default_cfgs["convnextv2_huge"]
    model_args = dict(in_channels=in_channels, num_classes=num_classes, depths=[3, 3, 27, 3],
                      dims=[352, 704, 1408, 2816], use_grn=True, layer_scale_init_value=0.0, **kwargs)
    return _create_convnext(pretrained, **dict(default_cfg=default_cfg, **model_args))
