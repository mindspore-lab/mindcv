"""
MindSpore implementation of `MLP-Mixer`.
Refer to MLP-Mixer: An all-MLP Architecture for Vision.
"""

import mindspore.nn as nn
import mindspore.ops as ops

from .helpers import load_pretrained
from .layers.compatibility import Dropout
from .registry import register_model

__all__ = [
    "MLPMixer",
    "mlp_mixer_s_p32",
    "mlp_mixer_s_p16",
    "mlp_mixer_b_p16",
    "mlp_mixer_b_p32",
    "mlp_mixer_l_p16",
    "mlp_mixer_l_p32",
    "mlp_mixer_h_p14"
]


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "first_conv": "to_patch_embedding.0",
        "classifier": "mlp_head",
        **kwargs,
    }


default_cfgs = {
    "mlp_mixer_s_p16": _cfg(url=""),
    "mlp_mixer_s_p32": _cfg(url=""),
    "mlp_mixer_b_p16": _cfg(url=""),
    "mlp_mixer_b_p32": _cfg(url=""),
    "mlp_mixer_l_p16": _cfg(url=""),
    "mlp_mixer_l_p32": _cfg(url=""),
    "mlp_mixer_h_p14": _cfg(url=""),
}


class FeedForward(nn.Cell):
    """Feed Forward Block. MLP Layer. FC -> GELU -> FC"""

    def __init__(self, dim, hidden_dim, dropout=0.):
        super(FeedForward, self).__init__()
        self.net = nn.SequentialCell(
            nn.Dense(dim, hidden_dim),
            nn.GELU(),
            Dropout(p=dropout),
            nn.Dense(hidden_dim, dim),
            Dropout(p=dropout)
        )

    def construct(self, x):
        return self.net(x)


class TransPose(nn.Cell):
    """TransPose Layer. Wrap operator Transpose for easy integration in nn.SequentialCell"""

    def __init__(self, permutation=(0, 2, 1), embedding=False):
        super(TransPose, self).__init__()
        self.permutation = permutation
        self.embedding = embedding
        if embedding:
            self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()

    def construct(self, x):
        if self.embedding:
            b, c, h, w = x.shape
            x = self.reshape(x, (b, c, h * w))
        x = self.transpose(x, self.permutation)
        return x


class MixerBlock(nn.Cell):
    """Mixer Layer with token-mixing MLP and channel-mixing MLP"""

    def __init__(self, n_patches, n_channels, token_dim, channel_dim, dropout=0.):
        super().__init__()
        self.token_mix = nn.SequentialCell(
            nn.LayerNorm((n_channels,)),
            TransPose((0, 2, 1)),
            FeedForward(n_patches, token_dim, dropout),
            TransPose((0, 2, 1))
        )
        self.channel_mix = nn.SequentialCell(
            nn.LayerNorm((n_channels,)),
            FeedForward(n_channels, channel_dim, dropout),
        )

    def construct(self, x):
        x = x + self.token_mix(x)
        x = x + self.channel_mix(x)
        return x


class MLPMixer(nn.Cell):
    r"""MLP-Mixer model class, based on
    `"MLP-Mixer: An all-MLP Architecture for Vision" <https://arxiv.org/abs/2105.01601>`_

    Args:
        depth (int) : number of MixerBlocks.
        patch_size (int or tuple) : size of a single image patch.
        n_patches (int) : number of patches.
        n_channels (int) : channels(dimension) of a single embedded patch.
        token_dim (int) : hidden dim of token-mixing MLP.
        channel_dim (int) : hidden dim of channel-mixing MLP.
        num_classes (int) : number of classification classes.
        in_channels: number the channels of the input. Default: 3.
    """

    def __init__(self, depth, patch_size, n_patches, n_channels, token_dim, channel_dim, num_classes=1000,
                 in_channels=3):
        super().__init__()
        self.n_patches = n_patches
        self.n_channels = n_channels
        # patch with shape of (3, patch_size, patch_size) is embedded to n_channels dim feature.
        self.to_patch_embedding = nn.SequentialCell(
            nn.Conv2d(in_channels, n_channels, patch_size, patch_size, pad_mode="pad", padding=0),
            TransPose(permutation=(0, 2, 1), embedding=True),
        )
        self.mixer_blocks = nn.SequentialCell()
        for _ in range(depth):
            self.mixer_blocks.append(MixerBlock(n_patches, n_channels, token_dim, channel_dim))
        self.layer_norm = nn.LayerNorm((n_channels,))
        self.mlp_head = nn.Dense(n_channels, num_classes)
        self.mean = ops.ReduceMean()
        self._initialize_weights()

    def construct(self, x):
        x = self.to_patch_embedding(x)
        x = self.mixer_blocks(x)
        x = self.layer_norm(x)
        x = self.mean(x, 1)
        return self.mlp_head(x)

    def _initialize_weights(self):
        # todo: implement weights init
        pass


def _check_resolution_and_length_of_patch(pr, sl):
    if isinstance(pr, int):
        ir = 224
        assert ir % pr == 0, 'Image resolution must be divisible by the patch resolution.'
        assert sl == (ir // pr) ** 2, "Sequence length must be equal to ir/pr."
    elif isinstance(pr, tuple) and list(map(type, pr)) == [int, int]:
        ir = (224, 224)
        assert ir[0] % pr[0] == 0 and ir[1] % pr[1] == 0, 'Image resolution must be divisible by the patch resolution.'
        assert sl == (ir[0] // pr[0]) * (ir[1] // pr[1]), "Sequence length must be equal to ir/pr."


@register_model
def mlp_mixer_s_p32(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    # number_of_layers, patch_resolution, length_of_sequence, hidden_size, mpl_dim_sequence, mpl_dim_channel
    nl, pr, ls, hs, ds, dc = 8, 32, 49, 512, 256, 2048
    _check_resolution_and_length_of_patch(pr, ls)
    model = MLPMixer(depth=nl, patch_size=pr, n_patches=ls, n_channels=hs,
                     token_dim=ds, channel_dim=dc, num_classes=num_classes, in_channels=in_channels, **kwargs)

    default_cfg = default_cfgs["mlp_mixer_s_p32"]
    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def mlp_mixer_s_p16(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    nl, pr, ls, hs, ds, dc = 8, 16, 196, 512, 256, 2048
    _check_resolution_and_length_of_patch(pr, ls)
    model = MLPMixer(depth=nl, patch_size=pr, n_patches=ls, n_channels=hs, token_dim=ds, channel_dim=dc,
                     num_classes=num_classes, in_channels=in_channels, **kwargs)

    default_cfg = default_cfgs["mlp_mixer_s_p16"]
    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def mlp_mixer_b_p32(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    nl, pr, ls, hs, ds, dc = 12, 32, 49, 768, 384, 3072
    _check_resolution_and_length_of_patch(pr, ls)
    model = MLPMixer(depth=nl, patch_size=pr, n_patches=ls, n_channels=hs, token_dim=ds, channel_dim=dc,
                     num_classes=num_classes, in_channels=in_channels, **kwargs)

    default_cfg = default_cfgs["mlp_mixer_b_p32"]
    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def mlp_mixer_b_p16(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    nl, pr, ls, hs, ds, dc = 12, 16, 196, 768, 384, 3072
    _check_resolution_and_length_of_patch(pr, ls)
    model = MLPMixer(depth=nl, patch_size=pr, n_patches=ls, n_channels=hs, token_dim=ds, channel_dim=dc,
                     num_classes=num_classes, in_channels=in_channels, **kwargs)

    default_cfg = default_cfgs["mlp_mixer_b_p16"]
    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def mlp_mixer_l_p32(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    nl, pr, ls, hs, ds, dc = 24, 32, 49, 1024, 512, 4096
    _check_resolution_and_length_of_patch(pr, ls)
    model = MLPMixer(depth=nl, patch_size=pr, n_patches=ls, n_channels=hs, token_dim=ds, channel_dim=dc,
                     num_classes=num_classes, in_channels=in_channels, **kwargs)

    default_cfg = default_cfgs["mlp_mixer_l_p32"]
    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def mlp_mixer_l_p16(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    nl, pr, ls, hs, ds, dc = 24, 16, 196, 1024, 512, 4096
    _check_resolution_and_length_of_patch(pr, ls)
    model = MLPMixer(depth=nl, patch_size=pr, n_patches=ls, n_channels=hs, token_dim=ds, channel_dim=dc,
                     num_classes=num_classes, in_channels=in_channels, **kwargs)

    default_cfg = default_cfgs["mlp_mixer_l_p16"]
    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def mlp_mixer_h_p14(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    nl, pr, ls, hs, ds, dc = 32, 14, 256, 1280, 640, 5120
    _check_resolution_and_length_of_patch(pr, ls)
    model = MLPMixer(depth=nl, patch_size=pr, n_patches=ls, n_channels=hs, token_dim=ds, channel_dim=dc,
                     num_classes=num_classes, in_channels=in_channels, **kwargs)

    default_cfg = default_cfgs["mlp_mixer_h_p14"]
    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model
