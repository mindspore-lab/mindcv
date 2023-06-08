"""
Implementation of the ConvMixer model.
Refer to "Patches Are All You Need?"
"""
import mindspore.nn as nn
from mindspore.ops import ReduceMean

from .registry import register_model
from .utils import load_pretrained

__all__ = [
    "convmixer_768_32",
    "convmixer_1024_20",
    "convmixer_1536_20"
]


def _cfg(classifier, url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "first_conv": 'network.0',
        "classifier": classifier,
        **kwargs
    }


default_cfgs = {
    "convmixer_768_32": _cfg(
                classifier="network.37",
                url="https://storage.googleapis.com/huawei-mindspore-hk/ConvMixer/Converted/convmixer_768_32.ckpt"),
    "convmixer_1024_20": _cfg(
                classifier="network.25",
                url="https://storage.googleapis.com/huawei-mindspore-hk/ConvMixer/Converted/convmixer_1024_20.ckpt"),
    "convmixer_1536_20": _cfg(
                classifier="network.25",
                url="https://storage.googleapis.com/huawei-mindspore-hk/ConvMixer/Converted/convmixer_1536_20.ckpt"),
}


class Residual(nn.Cell):
    """Residual connection. """

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def construct(self, *inputs, **kwargs):
        x = inputs[0]
        return self.fn(x) + x


class AvgPoolReduceMean(nn.Cell):
    """AvgPool cell implemented on the basis of ReduceMean op."""

    def construct(self, *inputs, **kwargs):
        """Forward pass."""
        x = inputs[0]
        return ReduceMean(True)(x, (2, 3))


class ConvMixer(nn.Cell):
    """ConvMixer model."""

    def __init__(
            self,
            dim,
            depth,
            kernel_size=9,
            patch_size=7,
            in_channels=3,
            n_classes=1000,
            act_type='gelu',
            onnx_export=False,
    ):
        super().__init__()
        if act_type.lower() == 'gelu':
            act = nn.GELU
        elif act_type.lower() == 'relu':
            act = nn.ReLU
        else:
            raise NotImplementedError()

        avg_pool = AvgPoolReduceMean() if onnx_export \
            else nn.AdaptiveAvgPool2d((1, 1))

        self.network = nn.SequentialCell(
            nn.Conv2d(
                in_channels,
                dim,
                kernel_size=patch_size,
                stride=patch_size,
                has_bias=True,
                pad_mode='pad',
                padding=0,
            ),
            act(),
            nn.BatchNorm2d(dim),
            *[nn.SequentialCell(
                Residual(
                    nn.SequentialCell(
                        nn.Conv2d(
                            dim,
                            dim,
                            kernel_size,
                            group=dim,
                            pad_mode='same',
                            has_bias=True
                        ),
                        act(),
                        nn.BatchNorm2d(dim)
                    )
                ),
                nn.Conv2d(
                    dim,
                    dim,
                    kernel_size=1,
                    has_bias=True,
                    pad_mode='pad',
                    padding=0,
                ),
                act(),
                nn.BatchNorm2d(dim)
            ) for _ in range(depth)],
            avg_pool,
            nn.Flatten(),
            nn.Dense(dim, n_classes)
        )

    def construct(self, *inputs, **kwargs):
        """Forward pass."""
        x = inputs[0]
        x = self.network(x)
        return x


@register_model
def convmixer_1536_20(pretrained: bool = False, num_classes=1000, in_channels=3, **kwargs):
    """Create ConvMixer-1536/20 model."""
    model = ConvMixer(
        1536,
        20,
        kernel_size=9,
        patch_size=7,
        in_channels=in_channels,
        n_classes=num_classes,
    )
    default_cfg = default_cfgs['convmixer_1536_20']

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def convmixer_1024_20(pretrained: bool = False, num_classes=1000, in_channels=3, **kwargs):
    """Create ConvMixer-1024/20 model."""
    model = ConvMixer(
        1024,
        20,
        kernel_size=9,
        patch_size=14,
        in_channels=in_channels,
        n_classes=num_classes,
    )
    default_cfg = default_cfgs['convmixer_1024_20']

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def convmixer_768_32(pretrained: bool = False, num_classes=1000, in_channels=3,
                     act_type='relu', **kwargs):
    """Create ConvMixer-768/32 model."""
    model = ConvMixer(
        768,
        32,
        kernel_size=7,
        patch_size=7,
        in_channels=in_channels,
        n_classes=num_classes,
        act_type=act_type,
    )
    default_cfg = default_cfgs['convmixer_768_32']

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model
