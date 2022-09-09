#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

from typing import Optional

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, Parameter
import mindspore.common.initializer as init

from .layers.drop_path import DropPath
from .utils import load_pretrained
from .registry import register_model

__all__ = [
    'ViT',
    'vit_b_16_224',
    'vit_b_32_224',
    'vit_l_16_224',
    'vit_l_32_224',
]


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000,
        'first_conv': '', 'classifier': '',
        **kwargs
    }


default_cfgs = {
    'vit_b_16_224': _cfg(url=''),
    'vit_b_32_224': _cfg(url=''),
    'vit_l_16_224': _cfg(url=''),
    'vit_l_32_224': _cfg(url='')
}


class PatchEmbedding(nn.Cell):

    def __init__(self,
                 image_size: int = 224,
                 patch_size: int = 16,
                 embed_dim: int = 768,
                 input_channels: int = 3) -> None:
        super(PatchEmbedding, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        self.conv = nn.Conv2d(input_channels, embed_dim, kernel_size=patch_size, stride=patch_size, has_bias=True)

    def construct(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        b, c, h, w = x.shape
        x = ops.reshape(x, (b, c, h * w))
        x = ops.transpose(x, (0, 2, 1))
        return x


class Attention(nn.Cell):

    def __init__(self,
                 dim: int,
                 num_heads: int = 8,
                 qkv_bias: bool = False,
                 qk_scale: Optional[float] = None,
                 drop_rate: float = 0.,
                 attention_drop_rate: float = 0.) -> None:
        super(Attention, self).__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads.'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or Tensor(head_dim ** -0.5)

        self.qkv = nn.Dense(dim, dim * 3, has_bias=qkv_bias)
        self.attn_drop = nn.Dropout(1 - attention_drop_rate)
        self.proj = nn.Dense(dim, dim)
        self.proj_drop = nn.Dropout(1 - drop_rate)
        self.softmax = nn.Softmax(axis=-1)

        self.attn_matmul_v = ops.BatchMatMul()
        self.q_matmul_k = ops.BatchMatMul(transpose_b=True)

    def construct(self, x: Tensor) -> Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x)
        qkv = ops.reshape(qkv, (b, n, 3, self.num_heads, c // self.num_heads))
        qkv = ops.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = ops.unstack(qkv, axis=0)

        attn = self.q_matmul_k(q, k)
        attn = ops.mul(attn, self.scale)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        out = self.attn_matmul_v(attn, v)
        out = ops.transpose(out, (0, 2, 1, 3))
        out = ops.reshape(out, (b, n, c))
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class FeedForward(nn.Cell):

    def __init__(self,
                 in_channels: int,
                 hidden_channels: Optional[int] = None,
                 out_channels: Optional[int] = None,
                 activation: nn.Cell = nn.GELU,
                 drop_rate: float = 0.) -> None:
        super(FeedForward, self).__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.fc1 = nn.Dense(in_channels, hidden_channels)
        self.activation = activation()
        self.fc2 = nn.Dense(hidden_channels, out_channels)
        self.dropout = nn.Dropout(1 - drop_rate)

    def construct(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class ResidualCell(nn.Cell):

    def __init__(self, cell: nn.Cell) -> None:
        super(ResidualCell, self).__init__()
        self.cell = cell

    def construct(self, x: Tensor) -> Tensor:
        return self.cell(x) + x


class TransformerEncoder(nn.Cell):

    def __init__(self,
                 dim: int,
                 num_layers: int,
                 num_heads: int,
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = False,
                 drop_rate: float = 0.,
                 attention_drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 activation: nn.Cell = nn.GELU,
                 norm: nn.Cell = nn.LayerNorm) -> None:
        super(TransformerEncoder, self).__init__()
        dpr = [i for i in ops.linspace(Tensor(0.), Tensor(drop_path_rate), num_layers)]

        layers = []
        for i in range(num_layers):
            normalization1 = norm((dim,))
            normalization2 = norm((dim,))
            attention = Attention(dim=dim,
                                  num_heads=num_heads,
                                  qkv_bias=qkv_bias,
                                  drop_rate=drop_rate,
                                  attention_drop_rate=attention_drop_rate)

            feedforward = FeedForward(in_channels=dim,
                                      hidden_channels=int(dim * mlp_ratio),
                                      activation=activation,
                                      drop_rate=drop_rate)

            if drop_path_rate > 0.:
                layers.append(
                    nn.SequentialCell([
                        ResidualCell(nn.SequentialCell([normalization1,
                                                        attention,
                                                        DropPath(dpr[i])])),
                        ResidualCell(nn.SequentialCell([normalization2,
                                                        feedforward,
                                                        DropPath(dpr[i])]))]))
            else:
                layers.append(
                    nn.SequentialCell([
                        ResidualCell(nn.SequentialCell([normalization1,
                                                        attention])),
                        ResidualCell(nn.SequentialCell([normalization2,
                                                        feedforward]))
                    ])
                )
        self.layers = nn.SequentialCell(layers)

    def construct(self, x: Tensor) -> Tensor:
        return self.layers(x)


class ViT(nn.Cell):

    def __init__(self,
                 image_size: int = 224,
                 in_channels: int = 3,
                 num_classes: int = 1000,
                 patch_size: int = 16,
                 embed_dim: int = 768,
                 num_layers: int = 12,
                 num_heads: int = 12,
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 drop_rate: float = 0.,
                 attention_drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 activation: nn.Cell = nn.GELU,
                 norm: Optional[nn.Cell] = nn.LayerNorm,
                 pool: str = 'cls') -> None:
        super(ViT, self).__init__()
        self.num_features = embed_dim
        self.patch_embedding = PatchEmbedding(image_size=image_size,
                                              patch_size=patch_size,
                                              embed_dim=embed_dim,
                                              input_channels=in_channels)
        num_patches = self.patch_embedding.num_patches

        if pool == "cls":
            self.cls_token = Parameter(
                init.initializer(init.Normal(sigma=1.0), (1, 1, embed_dim), ms.float32),
                name='cls',
                requires_grad=True
            )

            self.pos_embedding = Parameter(
                init.initializer(init.Normal(sigma=1.0), (1, num_patches + 1, embed_dim), ms.float32),
                name='pos_embedding',
                requires_grad=True
            )
        else:
            self.pos_embedding = Parameter(
                init.initializer(init.Normal(sigma=1.0), (1, num_patches, embed_dim), ms.float32),
                name='pos_embedding',
                requires_grad=True
            )

        self.pool = pool
        self.pos_dropout = nn.Dropout(1 - drop_rate)
        self.norm = norm((embed_dim,))
        self.transformer = TransformerEncoder(dim=embed_dim,
                                              num_layers=num_layers,
                                              num_heads=num_heads,
                                              mlp_ratio=mlp_ratio,
                                              qkv_bias=qkv_bias,
                                              drop_rate=drop_rate,
                                              attention_drop_rate=attention_drop_rate,
                                              drop_path_rate=drop_path_rate,
                                              activation=activation,
                                              norm=norm)

        self.classifier = nn.Dense(embed_dim, num_classes)

    def forward_features(self, x: Tensor) -> Tensor:

        x = self.patch_embedding(x)

        if self.pool == "cls":
            cls_tokens = ops.tile(self.cls_token, (x.shape[0], 1, 1))
            x = ops.concat((cls_tokens, x), axis=1)
            x += self.pos_embedding
        else:
            x += self.pos_embedding
        x = self.pos_dropout(x)
        x = self.transformer(x)
        x = self.norm(x)

        if self.pool == "cls":
            x = x[:, 0]
        else:
            x = ops.mean(x, (1, 2))  # (1,) or (1,2)

        return x

    def forward_head(self, x: Tensor) -> Tensor:
        x = self.classifier(x)
        return x

    def construct(self, x: Tensor) -> Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


@register_model
def vit_b_16_224(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs['vit_b_16_224']
    model = ViT(in_channels=in_channels,
                num_classes=num_classes,
                patch_size=16,
                embed_dim=768,
                num_layers=12,
                num_heads=12,
                mlp_ratio=4.0,
                **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def vit_b_32_224(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs['vit_b_32_224']
    model = ViT(in_channels=in_channels,
                num_classes=num_classes,
                patch_size=32,
                embed_dim=768,
                num_layers=12,
                num_heads=12,
                mlp_ratio=4.0,
                **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def vit_l_16_224(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs['vit_l_16_224']
    model = ViT(in_channels=in_channels,
                num_classes=num_classes,
                patch_size=16,
                embed_dim=1024,
                num_layers=24,
                num_heads=16,
                mlp_ratio=4.0,
                **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def vit_l_32_224(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs['vit_l_32_224']
    model = ViT(in_channels=in_channels,
                num_classes=num_classes,
                patch_size=32,
                embed_dim=1024,
                num_layers=24,
                num_heads=16,
                mlp_ratio=4.0,
                **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model
