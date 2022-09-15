import numpy as np
from typing import List, Tuple, Optional

from mindspore import Parameter, Tensor, nn, ops
from mindspore import dtype as mstype
from mindspore.common import initializer as weight_init

from .utils import load_pretrained
from .registry import register_model


__all__ = [
    'ConvNeXt',
    'convnext_tiny',
    'convnext_small',
    'convnext_base',
    'convnext_large',
    'convnext_xlarge'
]


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000,
        'first_conv': '', 'classifier': '',
        **kwargs
    }


default_cfgs = {
    'convnext_tiny': _cfg(url=''),
    'convnext_small': _cfg(url=''),
    'convnext_base': _cfg(url=''),
    'convnext_large': _cfg(url=''),
    'convnext_xlarge': _cfg(url=''),
}


class Identity(nn.Cell):
    """Identity"""

    def construct(self, x):
        return x


class DropPath(nn.Cell):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self,
                 drop_prob: float,
                 ndim: int
                 ) -> None:
        super(DropPath, self).__init__()
        self.drop = nn.Dropout(keep_prob=1 - drop_prob)
        shape = (1,) + (1,) * (ndim + 1)
        self.ndim = ndim
        self.mask = Tensor(np.ones(shape), dtype=mstype.float32)

    def construct(self, x: Tensor) -> Tensor:
        if not self.training:
            return x
        mask = ops.tile(self.mask, (x.shape[0],) + (1,) * (self.ndim + 1))
        out = self.drop(mask)
        out = out * x
        return out


class DropPath2D(DropPath):
    """DropPath2D"""

    def __init__(self,
                 drop_prob: float
                 ) -> None:
        super(DropPath2D, self).__init__(drop_prob=drop_prob, ndim=2)


class ConvNextLayerNorm(nn.LayerNorm):
    """ConvNextLayerNorm"""

    def __init__(self,
                 normalized_shape: Tuple[int],
                 epsilon: float,
                 norm_axis: int = -1
                 ) -> None:
        super(ConvNextLayerNorm, self).__init__(normalized_shape=normalized_shape, epsilon=epsilon)
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
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Dense -> GELU -> Dense; Permute back
    We use (2) as we find it slightly faster in PyTorch
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self,
                 dim: int,
                 drop_path: float = 0.,
                 layer_scale_init_value: float = 1e-6
                 ) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, group=dim, has_bias=True)  # depthwise conv
        self.norm = ConvNextLayerNorm((dim,), epsilon=1e-6)
        self.pwconv1 = nn.Dense(dim, 4 * dim)  # pointwise/1x1 convs, implemented with Dense layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Dense(4 * dim, dim)
        self.gamma_ = Parameter(Tensor(layer_scale_init_value * np.ones((dim)), dtype=mstype.float32),
                               requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath2D(drop_path) if drop_path > 0. else Identity()

    def construct(self, x: Tensor) -> Tensor:
        """Block construct"""
        downsample = x
        x = self.dwconv(x)
        x = ops.transpose(x, (0, 2, 3, 1))
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma_ is not None:
            x = self.gamma_ * x
        x = ops.transpose(x, (0, 3, 1, 2))
        x = downsample + self.drop_path(x)
        return x


class ConvNeXt(nn.Cell):

    def __init__(self,
                 in_channel: int,
                 num_classes: int,
                 depths: List[int],
                 dims: List[int],
                 drop_path_rate: float = 0.,
                 layer_scale_init_value: float = 1e-6,
                 head_init_scale: float = 1.):
        super().__init__()

        self.downsample_layers = nn.CellList()  # stem and 3 intermediate down_sampling conv layers
        stem = nn.SequentialCell(
            nn.Conv2d(in_channel, dims[0], kernel_size=4, stride=4, has_bias=True),
            ConvNextLayerNorm((dims[0],), epsilon=1e-6, norm_axis=1)
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.SequentialCell(
                ConvNextLayerNorm((dims[i],), epsilon=1e-6, norm_axis=1),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2, has_bias=True),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.CellList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x for x in np.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            blocks = []
            for j in range(depths[i]):
                blocks.append(Block(dim=dims[i], drop_path=dp_rates[cur + j],
                                    layer_scale_init_value=layer_scale_init_value))
            stage = nn.SequentialCell(blocks)
            self.stages.append(stage)
            cur += depths[i]

        self.norm = ConvNextLayerNorm((dims[-1],), epsilon=1e-6)  # final norm layer
        self.classifier = nn.Dense(dims[-1], num_classes) # classifier

        self.init_weights()
        self.classifier.weight.set_data(self.classifier.weight * head_init_scale)
        self.classifier.bias.set_data(self.classifier.bias * head_init_scale)

    def init_weights(self) -> None:
        """init_weights"""
        for _, cell in self.cells_and_names():
            if isinstance(cell, (nn.Dense, nn.Conv2d)):
                cell.weight.set_data(weight_init.initializer(weight_init.TruncatedNormal(sigma=0.02),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))
                if isinstance(cell, nn.Dense) and cell.bias is not None:
                    cell.bias.set_data(weight_init.initializer(weight_init.Zero(),
                                                               cell.bias.shape,
                                                               cell.bias.dtype))

    def forward_head(self, x: Tensor) -> Tensor:
        x = self.classifier(x)
        return x

    def forward_features(self, x: Tensor) -> Tensor:
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)

    def construct(self, x: Tensor) -> Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


@register_model
def convnext_tiny(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> ConvNeXt:
    default_cfg = default_cfgs['convnext_tiny']
    model = ConvNeXt(in_channel=in_channels, num_classes=num_classes, depths=[3, 3, 9, 3],
                     dims=[96, 192, 384, 768],
                     **kwargs)
    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def convnext_small(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> ConvNeXt:
    default_cfg = default_cfgs['convnext_small']
    model = ConvNeXt(in_channel=in_channels, num_classes=num_classes, depths=[3, 3, 27, 3],
                     dims=[96, 192, 384, 768],
                     **kwargs)
    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def convnext_base(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> ConvNeXt:
    default_cfg = default_cfgs['convnext_base']
    model = ConvNeXt(in_channel=in_channels, num_classes=num_classes, depths=[3, 3, 27, 3],
                     dims=[128, 256, 512, 1024],
                     **kwargs)
    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def convnext_large(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> ConvNeXt:
    default_cfg = default_cfgs['convnext_large']
    model = ConvNeXt(in_channel=in_channels, num_classes=num_classes, depths=[3, 3, 27, 3],
                     dims=[192, 384, 768, 1536],
                     **kwargs)
    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def convnext_xlarge(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> ConvNeXt:
    default_cfg = default_cfgs['convnext_xlarge']
    model = ConvNeXt(in_channel=in_channels, num_classes=num_classes, depths=[3, 3, 27, 3],
                     dims=[256, 512, 1024, 2048],
                     **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model
