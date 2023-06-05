"""
MindSpore implementation of `MobileViT`.
Refer to MobileViT：Light-weight, General-purpose, and Mobile-friendly Vision Transformer.
"""

import math
from typing import Dict, Optional, Tuple, Union

import mindspore.common.initializer as init
from mindspore import Tensor, nn, ops

from .helpers import load_pretrained, make_divisible
from .layers.compatibility import Dropout, Interpolate
from .layers.pooling import GlobalAvgPooling
from .registry import register_model

__all__ = [
    "mobilevit_xx_small",
    "mobilevit_x_small",
    "mobilevit_small",
]


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "first_conv": "conv_1.features.0",
        "classifier": "classifier.3",
        **kwargs,
    }


default_cfgs = {
    "mobilevit_xx_small": _cfg(
        url="https://download.mindspore.cn/toolkits/mindcv/mobilevit/mobilevit_xx_small-af9da8a0.ckpt"),
    "mobilevit_x_small": _cfg(
        url="https://download.mindspore.cn/toolkits/mindcv/mobilevit/mobilevit_x_small-673fc6f2.ckpt"),
    "mobilevit_small": _cfg(
        url="https://download.mindspore.cn/toolkits/mindcv/mobilevit/mobilevit_small-caf79638.ckpt"),
}


class ConvLayer(nn.Cell):
    """ Conv2d + BN + Act"""
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 pad_mode: str = "pad",
                 padding: Optional[int] = None,
                 dilation: int = 1,
                 groups: int = 1,
                 norm: Optional[nn.Cell] = nn.BatchNorm2d,
                 activation: Optional[nn.Cell] = nn.SiLU,
                 has_bias: Optional[bool] = False) -> None:
        super().__init__()

        if pad_mode == "pad":
            if padding is None:
                padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
        else:
            padding = 0

        if has_bias is None:
            has_bias = norm is None

        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                pad_mode=pad_mode,
                padding=padding,
                dilation=dilation,
                group=groups,
                has_bias=has_bias)
        ]

        if norm:
            layers.append(norm(out_channels, momentum=0.9))
        if activation:
            layers.append(activation())

        self.features = nn.SequentialCell(layers)

    def construct(self, x):
        output = self.features(x)
        return output


class InvertedResidual(nn.Cell):
    """
    This class implements the inverted residual block, as described in
    `MobileNetv2 <https://arxiv.org/abs/1801.04381>`_ paper

    Args:
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H_{in}, W_{in})`
        out_channels (int): :math:`C_{out}` from an expected output of size :math:`(N, C_{out}, H_{out}, W_{out)`
        stride (int): Use convolutions with a stride. Default: 1
        expand_ratio (Union[int, float]): Expand the input channels by this factor in depth-wise conv
        skip_connection (Optional[bool]): Use skip-connection. Default: True

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})`

    .. note::
        If `in_channels =! out_channels` and `stride > 1`, we set `skip_connection=False`

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: Union[int, float],
        skip_connection: Optional[bool] = True,
    ) -> None:
        assert stride in [1, 2]
        hidden_dim = make_divisible(int(round(in_channels * expand_ratio)), 8)

        super().__init__()

        block = []
        if expand_ratio != 1:
            block.append(
                ConvLayer(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    kernel_size=1,
                    norm=nn.BatchNorm2d,
                    activation=nn.SiLU
                ),
            )

        block.append(
            ConvLayer(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=3,
                stride=stride,
                groups=hidden_dim,
                norm=nn.BatchNorm2d,
                activation=nn.SiLU
            ),
        )

        block.append(
           ConvLayer(
                in_channels=hidden_dim,
                out_channels=out_channels,
                kernel_size=1,
                has_bias=False,
                activation=None
            ),
        )

        self.block = nn.SequentialCell(block)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.exp = expand_ratio
        self.stride = stride
        self.use_res_connect = (
            self.stride == 1 and in_channels == out_channels and skip_connection
        )

    def construct(self, x: Tensor, *args, **kwargs) -> Tensor:
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)


class MultiHeadAttention(nn.Cell):
    """
    This layer applies a multi-head self- or cross-attention as described in
    `Attention is all you need <https://arxiv.org/abs/1706.03762>`_ paper

    Args:
        embed_dim (int): :math:`C_{in}` from an expected input of size :math:`(N, P, C_{in})`
        num_heads (int): Number of heads in multi-head attention
        attn_dropout (float): Attention dropout. Default: 0.0
        bias (bool): Use bias or not. Default: ``True``

    Shape:
        - Input: :math:`(N, P, C_{in})` where :math:`N` is batch size, :math:`P` is number of patches,
        and :math:`C_{in}` is input embedding dim
        - Output: same shape as the input

    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attn_dropout: float = 0.0,
        bias: bool = True,
        *args,
        **kwargs
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                "Embedding dim must be divisible by number of heads in {}. Got: embed_dim={} and num_heads={}".format(
                    self.__class__.__name__, embed_dim, num_heads
                )
            )

        self.qkv_proj = nn.Dense(in_channels=embed_dim, out_channels=embed_dim * 3, has_bias=bias)

        self.attn_dropout = Dropout(p=attn_dropout)
        self.out_proj = nn.Dense(in_channels=embed_dim, out_channels=embed_dim, has_bias=bias)

        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.softmax = nn.Softmax(axis=-1)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.batch_matmul = ops.BatchMatMul()

    def construct(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv_proj(x)
        qkv = ops.reshape(qkv, (B, N, 3, self.num_heads, C // self.num_heads))
        qkv = ops.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = ops.BatchMatMul(transpose_b=True)(q, k) * self.scaling
        attn = nn.Softmax(axis=-1)(attn)
        attn = self.attn_dropout(attn)

        x = ops.transpose(ops.BatchMatMul()(attn, v), (0, 2, 1, 3))
        x = ops.reshape(x, (B, N, C))
        x = self.out_proj(x)
        return x


class TransformerEncoder(nn.Cell):
    """
    This class defines the pre-norm `Transformer encoder <https://arxiv.org/abs/1706.03762>`_
    Args:
        embed_dim (int): :math:`C_{in}` from an expected input of size :math:`(N, P, C_{in})`
        ffn_latent_dim (int): Inner dimension of the FFN
        num_heads (int) : Number of heads in multi-head attention. Default: 8
        attn_dropout (float): Dropout rate for attention in multi-head attention. Default: 0.0
        dropout (float): Dropout rate. Default: 0.0
        ffn_dropout (float): Dropout between FFN layers. Default: 0.0

    Shape:
        - Input: :math:`(N, P, C_{in})` where :math:`N` is batch size, :math:`P` is number of patches,
        and :math:`C_{in}` is input embedding dim
        - Output: same shape as the input
    """

    def __init__(
        self,
        embed_dim: int,
        ffn_latent_dim: int,
        num_heads: Optional[int] = 8,
        attn_dropout: Optional[float] = 0.0,
        dropout: Optional[float] = 0.0,
        ffn_dropout: Optional[float] = 0.0,
        *args,
        **kwargs
    ) -> None:

        super().__init__()

        attn_unit = MultiHeadAttention(
            embed_dim,
            num_heads,
            attn_dropout=attn_dropout,
            bias=True
        )

        self.pre_norm_mha = nn.SequentialCell(
            nn.LayerNorm((embed_dim,)),
            attn_unit,
            Dropout(p=dropout)
        )

        self.pre_norm_ffn = nn.SequentialCell(
            nn.LayerNorm((embed_dim,)),
            nn.Dense(in_channels=embed_dim, out_channels=ffn_latent_dim, has_bias=True),
            nn.SiLU(),
            Dropout(p=ffn_dropout),
            nn.Dense(in_channels=ffn_latent_dim, out_channels=embed_dim, has_bias=True),
            Dropout(p=dropout)
        )
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_latent_dim
        self.ffn_dropout = ffn_dropout
        self.std_dropout = dropout

    def construct(self, x: Tensor) -> Tensor:
        # multi-head attention
        res = x
        x = self.pre_norm_mha(x)
        x = x + res

        # feed forward network
        x = x + self.pre_norm_ffn(x)
        return x


class MobileViTBlock(nn.Cell):
    """
    This class defines the `MobileViT block <https://arxiv.org/abs/2110.02178?context=cs.LG>`_

    Args:
        opts: command line arguments
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H, W)`
        transformer_dim (int): Input dimension to the transformer unit
        ffn_dim (int): Dimension of the FFN block
        n_transformer_blocks (int): Number of transformer blocks. Default: 2
        head_dim (int): Head dimension in the multi-head attention. Default: 32
        attn_dropout (float): Dropout in multi-head attention. Default: 0.0
        dropout (float): Dropout rate. Default: 0.0
        ffn_dropout (float): Dropout between FFN layers in transformer. Default: 0.0
        patch_h (int): Patch height for unfolding operation. Default: 8
        patch_w (int): Patch width for unfolding operation. Default: 8
        transformer_norm_layer (Optional[str]): Normalization layer in the transformer block. Default: layer_norm
        conv_ksize (int): Kernel size to learn local representations in MobileViT block. Default: 3
        no_fusion (Optional[bool]): Do not combine the input and output feature maps. Default: False
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        transformer_dim: int,
        ffn_dim: int,
        n_transformer_blocks: int = 2,
        head_dim: int = 32,
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        patch_h: int = 8,
        patch_w: int = 8,
        conv_ksize: Optional[int] = 3,
        *args,
        **kwargs
    ) -> None:
        super().__init__()

        conv_3x3_in = ConvLayer(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=conv_ksize,
            stride=1
        )
        conv_1x1_in = ConvLayer(
            in_channels=in_channels,
            out_channels=transformer_dim,
            kernel_size=1,
            stride=1
        )

        conv_1x1_out = ConvLayer(
            in_channels=transformer_dim,
            out_channels=in_channels,
            kernel_size=1,
            stride=1
        )
        conv_3x3_out = ConvLayer(
            in_channels=2 * in_channels,
            out_channels=out_channels,
            kernel_size=conv_ksize,
            stride=1,
            pad_mode="pad",
            padding=1
        )

        local_rep = []
        local_rep.append(conv_3x3_in)
        local_rep.append(conv_1x1_in)
        self.local_rep = nn.SequentialCell(local_rep)

        assert transformer_dim % head_dim == 0
        num_heads = transformer_dim // head_dim

        self.global_rep = [
            TransformerEncoder(
                embed_dim=transformer_dim,
                ffn_latent_dim=ffn_dim,
                num_heads=num_heads,
                attn_dropout=attn_dropout,
                dropout=dropout,
                ffn_dropout=ffn_dropout
            )
            for _ in range(n_transformer_blocks)
        ]
        self.global_rep.append(nn.LayerNorm((transformer_dim,)))
        self.global_rep = nn.CellList(self.global_rep)

        self.conv_proj = conv_1x1_out
        self.fusion = conv_3x3_out

        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_area = self.patch_w * self.patch_h

        self.cnn_in_dim = in_channels
        self.cnn_out_dim = transformer_dim
        self.n_heads = num_heads
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.ffn_dropout = ffn_dropout
        self.n_blocks = n_transformer_blocks
        self.conv_ksize = conv_ksize
        self.interpolate = Interpolate(mode="bilinear", align_corners=True)

    def unfolding(self, x: Tensor) -> Tuple[Tensor, Dict]:
        patch_w, patch_h = self.patch_w, self.patch_h
        patch_area = patch_w * patch_h
        batch_size, in_channels, orig_h, orig_w = x.shape

        new_h = int(math.ceil(orig_h / self.patch_h) * self.patch_h)
        new_w = int(math.ceil(orig_w / self.patch_w) * self.patch_w)

        interpolate = False
        if new_w != orig_w or new_h != orig_h:
            # Note: Padding can be done, but then it needs to be handled in attention function.
            x = self.interpolate(x, size=(new_h, new_w))
            interpolate = True

        # number of patches along width and height
        num_patch_w = new_w // patch_w  # n_w
        num_patch_h = new_h // patch_h  # n_h
        num_patches = num_patch_h * num_patch_w  # N

        # [B, C, H, W] -> [B * C * n_h, p_h, n_w, p_w]
        x = ops.reshape(x, (batch_size * in_channels * num_patch_h, patch_h, num_patch_w, patch_w))
        # [B * C * n_h, p_h, n_w, p_w] -> [B * C * n_h, n_w, p_h, p_w]
        x = ops.transpose(x, (0, 2, 1, 3))
        # [B * C * n_h, n_w, p_h, p_w] -> [B, C, N, P] where P = p_h * p_w and N = n_h * n_w
        x = ops.reshape(x, (batch_size, in_channels, num_patches, patch_area))
        # [B, C, N, P] -> [B, P, N, C]
        x = ops.transpose(x, (0, 3, 2, 1))
        # [B, P, N, C] -> [BP, N, C]
        x = ops.reshape(x, (batch_size * patch_area, num_patches, -1))

        info_dict = {
            "orig_size": (orig_h, orig_w),
            "batch_size": batch_size,
            "interpolate": interpolate,
            "total_patches": num_patches,
            "num_patches_w": num_patch_w,
            "num_patches_h": num_patch_h,
        }

        return x, info_dict

    def folding(self, x: Tensor, info_dict: Dict) -> Tensor:
        n_dim = ops.rank(x)
        assert n_dim == 3, "Tensor should be of shape BPxNxC. Got: {}".format(
            x.shape
        )
        # [BP, N, C] --> [B, P, N, C]
        x = x.view(
            info_dict["batch_size"], self.patch_area, info_dict["total_patches"], -1
        )

        batch_size, pixels, num_patches, channels = x.shape
        num_patch_h = info_dict["num_patches_h"]
        num_patch_w = info_dict["num_patches_w"]

        # [B, P, N, C] -> [B, C, N, P]
        x = ops.transpose(x, (0, 3, 2, 1))
        # [B, C, N, P] -> [B*C*n_h, n_w, p_h, p_w]
        x = ops.reshape(x, (batch_size * channels * num_patch_h, num_patch_w, self.patch_h, self.patch_w))
        # [B*C*n_h, n_w, p_h, p_w] -> [B*C*n_h, p_h, n_w, p_w]
        x = ops.transpose(x, (0, 2, 1, 3))
        # [B*C*n_h, p_h, n_w, p_w] -> [B, C, H, W]
        x = ops.reshape(x, (batch_size, channels, num_patch_h * self.patch_h, num_patch_w * self.patch_w))
        if info_dict["interpolate"]:
            x = self.interpolate(x, size=info_dict["orig_size"])
        return x

    def construct(self, x: Tensor) -> Tensor:
        res = x
        fm = self.local_rep(x)
        # convert feature map to patches
        patches, info_dict = self.unfolding(fm)
        # learn global representations
        for transformer_layer in self.global_rep:
            patches = transformer_layer(patches)
        # [B x Patch x Patches x C] -> [B x C x Patches x Patch]
        fm = self.folding(x=patches, info_dict=info_dict)
        fm = self.conv_proj(fm)
        fm = self.fusion(ops.concat((res, fm), 1))
        return fm


class MobileViT(nn.Cell):
    """
    This class implements the `MobileViT architecture <https://arxiv.org/abs/2110.02178?context=cs.LG>`_
    """
    def __init__(self, model_cfg: Dict, num_classes: int = 1000):
        super().__init__()

        image_channels = 3
        out_channels = 16

        self.conv_1 = ConvLayer(
            in_channels=image_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2
        )

        self.layer_1, out_channels = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer1"])
        self.layer_2, out_channels = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer2"])
        self.layer_3, out_channels = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer3"])
        self.layer_4, out_channels = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer4"])
        self.layer_5, out_channels = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer5"])

        exp_channels = min(model_cfg["last_layer_exp_factor"] * out_channels, 960)
        self.conv_1x1_exp = ConvLayer(
            in_channels=out_channels,
            out_channels=exp_channels,
            kernel_size=1
        )

        classifier = []
        classifier.append(GlobalAvgPooling())
        classifier.append(nn.Flatten())
        if 0.0 < model_cfg["cls_dropout"] < 1.0:
            classifier.append(Dropout(p=model_cfg["cls_dropout"]))
        classifier.append(nn.Dense(in_channels=exp_channels, out_channels=num_classes))
        self.classifier = nn.SequentialCell(classifier)
        self._initialize_weights()

    def _make_layer(self, input_channel, cfg: Dict) -> Tuple[nn.SequentialCell, int]:
        block_type = cfg.get("block_type", "mobilevit")
        if block_type.lower() == "mobilevit":
            return self._make_mit_layer(input_channel=input_channel, cfg=cfg)
        else:
            return self._make_mobilenet_layer(input_channel=input_channel, cfg=cfg)

    @staticmethod
    def _make_mobilenet_layer(input_channel: int, cfg: Dict) -> Tuple[nn.SequentialCell, int]:
        output_channels = cfg.get("out_channels")
        num_blocks = cfg.get("num_blocks", 2)
        expand_ratio = cfg.get("expand_ratio", 4)
        block = []

        for i in range(num_blocks):
            stride = cfg.get("stride", 1) if i == 0 else 1

            layer = InvertedResidual(
                in_channels=input_channel,
                out_channels=output_channels,
                stride=stride,
                expand_ratio=expand_ratio
            )
            block.append(layer)
            input_channel = output_channels

        return nn.SequentialCell(block), input_channel

    @staticmethod
    def _make_mit_layer(input_channel: int, cfg: Dict) -> [nn.SequentialCell, int]:
        stride = cfg.get("stride", 1)
        block = []

        if stride == 2:
            layer = InvertedResidual(
                in_channels=input_channel,
                out_channels=cfg.get("out_channels"),
                stride=stride,
                expand_ratio=cfg.get("mv_expand_ratio", 4)
            )

            block.append(layer)
            input_channel = cfg.get("out_channels")

        transformer_dim = cfg["transformer_channels"]
        ffn_dim = cfg.get("ffn_dim")
        num_heads = cfg.get("num_heads", 4)
        head_dim = transformer_dim // num_heads

        if transformer_dim % head_dim != 0:
            raise ValueError("Transformer input dimension should be divisible by head dimension. "
                             "Got {} and {}.".format(transformer_dim, head_dim))

        block.append(MobileViTBlock(
            in_channels=input_channel,
            out_channels=cfg.get("out_channels"),
            transformer_dim=transformer_dim,
            ffn_dim=ffn_dim,
            n_transformer_blocks=cfg.get("transformer_blocks", 1),
            patch_h=cfg.get("patch_h", 2),
            patch_w=cfg.get("patch_w", 2),
            dropout=cfg.get("dropout", 0.1),
            ffn_dropout=cfg.get("ffn_dropout", 0.0),
            attn_dropout=cfg.get("attn_dropout", 0.1),
            head_dim=head_dim,
            conv_ksize=3
        ))

        return nn.SequentialCell(block), input_channel

    def _initialize_weights(self) -> None:
        """Initialize weights for cells."""
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Dense):
                if cell.weight is not None:
                    cell.weight.set_data(init.initializer(init.TruncatedNormal(sigma=.02), cell.weight.shape,
                                                          cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(init.initializer('zeros', cell.bias.shape, cell.bias.dtype))
            elif isinstance(cell, (nn.LayerNorm, nn.BatchNorm2d)):
                if cell.gamma is not None:
                    cell.gamma.set_data(init.initializer('ones', cell.gamma.shape, cell.gamma.dtype))
                if cell.beta is not None:
                    cell.beta.set_data(init.initializer('zeros', cell.beta.shape, cell.beta.dtype))
            elif isinstance(cell, nn.Conv2d):
                if cell.weight is not None:
                    cell.weight.set_data(init.initializer(init.HeNormal(mode='fan_out', nonlinearity='leaky_relu'),
                                         cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(init.initializer('zeros', cell.bias.shape, cell.bias.dtype))

    def construct(self, x: Tensor) -> Tensor:
        x = self.conv_1(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.conv_1x1_exp(x)
        x = self.classifier(x)
        return x


def get_config(mode: str = "xxs") -> dict:
    if mode == "xx_small":
        mv2_exp_mult = 2
        config = {
            "layer1": {
                "out_channels": 16,
                "expand_ratio": mv2_exp_mult,
                "num_blocks": 1,
                "stride": 1,
                "block_type": "mv2",
            },
            "layer2": {
                "out_channels": 24,
                "expand_ratio": mv2_exp_mult,
                "num_blocks": 3,
                "stride": 2,
                "block_type": "mv2",
            },
            "layer3": {  # 28x28
                "out_channels": 48,
                "transformer_channels": 64,
                "ffn_dim": 128,
                "transformer_blocks": 2,
                "patch_h": 2,  # 8,
                "patch_w": 2,  # 8,
                "stride": 2,
                "mv_expand_ratio": mv2_exp_mult,
                "num_heads": 4,
                "block_type": "mobilevit",
            },
            "layer4": {  # 14x14
                "out_channels": 64,
                "transformer_channels": 80,
                "ffn_dim": 160,
                "transformer_blocks": 4,
                "patch_h": 2,  # 4,
                "patch_w": 2,  # 4,
                "stride": 2,
                "mv_expand_ratio": mv2_exp_mult,
                "num_heads": 4,
                "block_type": "mobilevit",
            },
            "layer5": {  # 7x7
                "out_channels": 80,
                "transformer_channels": 96,
                "ffn_dim": 192,
                "transformer_blocks": 3,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mv_expand_ratio": mv2_exp_mult,
                "num_heads": 4,
                "block_type": "mobilevit",
            },
            "last_layer_exp_factor": 4,
            "cls_dropout": 0.1
        }
    elif mode == "x_small":
        mv2_exp_mult = 4
        config = {
            "layer1": {
                "out_channels": 32,
                "expand_ratio": mv2_exp_mult,
                "num_blocks": 1,
                "stride": 1,
                "block_type": "mv2",
            },
            "layer2": {
                "out_channels": 48,
                "expand_ratio": mv2_exp_mult,
                "num_blocks": 3,
                "stride": 2,
                "block_type": "mv2",
            },
            "layer3": {  # 28x28
                "out_channels": 64,
                "transformer_channels": 96,
                "ffn_dim": 192,
                "transformer_blocks": 2,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mv_expand_ratio": mv2_exp_mult,
                "num_heads": 4,
                "block_type": "mobilevit",
            },
            "layer4": {  # 14x14
                "out_channels": 80,
                "transformer_channels": 120,
                "ffn_dim": 240,
                "transformer_blocks": 4,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mv_expand_ratio": mv2_exp_mult,
                "num_heads": 4,
                "block_type": "mobilevit",
            },
            "layer5": {  # 7x7
                "out_channels": 96,
                "transformer_channels": 144,
                "ffn_dim": 288,
                "transformer_blocks": 3,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mv_expand_ratio": mv2_exp_mult,
                "num_heads": 4,
                "block_type": "mobilevit",
            },
            "last_layer_exp_factor": 4,
            "cls_dropout": 0.1
        }
    elif mode == "small":
        mv2_exp_mult = 4
        config = {
            "layer1": {
                "out_channels": 32,
                "expand_ratio": mv2_exp_mult,
                "num_blocks": 1,
                "stride": 1,
                "block_type": "mv2",
            },
            "layer2": {
                "out_channels": 64,
                "expand_ratio": mv2_exp_mult,
                "num_blocks": 3,
                "stride": 2,
                "block_type": "mv2",
            },
            "layer3": {  # 28x28
                "out_channels": 96,
                "transformer_channels": 144,
                "ffn_dim": 288,
                "transformer_blocks": 2,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mv_expand_ratio": mv2_exp_mult,
                "num_heads": 4,
                "block_type": "mobilevit",
            },
            "layer4": {  # 14x14
                "out_channels": 128,
                "transformer_channels": 192,
                "ffn_dim": 384,
                "transformer_blocks": 4,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mv_expand_ratio": mv2_exp_mult,
                "num_heads": 4,
                "block_type": "mobilevit",
            },
            "layer5": {  # 7x7
                "out_channels": 160,
                "transformer_channels": 240,
                "ffn_dim": 480,
                "transformer_blocks": 3,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mv_expand_ratio": mv2_exp_mult,
                "num_heads": 4,
                "block_type": "mobilevit",
            },
            "last_layer_exp_factor": 4,
            "cls_dropout": 0.1
        }
    else:
        raise NotImplementedError

    for k in ["layer1", "layer2", "layer3", "layer4", "layer5"]:
        config[k].update({"dropout": 0.1, "ffn_dropout": 0.0, "attn_dropout": 0.0})

    return config


@register_model
def mobilevit_xx_small(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> MobileViT:
    config = get_config("xx_small")
    model = MobileViT(config, num_classes=num_classes, **kwargs)
    default_cfg = default_cfgs["mobilevit_xx_small"]
    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def mobilevit_x_small(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> MobileViT:
    config = get_config("x_small")
    model = MobileViT(config, num_classes=num_classes, **kwargs)
    default_cfg = default_cfgs["mobilevit_x_small"]
    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def mobilevit_small(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> MobileViT:
    config = get_config("small")
    model = MobileViT(config, num_classes=num_classes, **kwargs)
    default_cfg = default_cfgs["mobilevit_small"]
    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model
