"""Reference:https://github.com/apple/ml-fastvit"""
import copy
import math
import os
from collections import OrderedDict
from functools import partial
from typing import List, Optional, Tuple, Union

import mindspore as ms
import mindspore.common.initializer as init
from mindspore import nn, ops
from mindspore.numpy import ones

from mindcv.models.layers.pooling import GlobalAvgPooling
from mindcv.models.registry import register_model

IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 256, 256),
        "pool_size": None,
        "crop_pct": 0.95,
        "interpolation": "bicubic",
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "classifier": "head",
        **kwargs,
    }


default_cfgs = {
    "fastvit_t": _cfg(crop_pct=0.9),
    "fastvit_s": _cfg(crop_pct=0.9),
    "fastvit_m": _cfg(crop_pct=0.95),
}


def convolutional_stem(
    in_channels: int, out_channels: int, inference_mode: bool = False
) -> nn.SequentialCell:
    """Build convolutional stem with MobileOne blocks.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        inference_mode: Flag to instantiate model in inference mode. Default: ``False``

    Returns:
        nn.Sequential object with stem elements.
    """
    return nn.SequentialCell(
        MobileOneBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            group=1,
            inference_mode=inference_mode,
            use_se=False,
            num_conv_branches=1,
        ),
        MobileOneBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            group=out_channels,
            inference_mode=inference_mode,
            use_se=False,
            num_conv_branches=1,
        ),
        MobileOneBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            group=1,
            inference_mode=inference_mode,
            use_se=False,
            num_conv_branches=1,
        ),
    )


class MHSA(nn.Cell):
    """Multi-headed Self Attention module.

    Source modified from:
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    """

    def __init__(
        self,
        dim: int,
        head_dim: int = 32,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        """Build MHSA module that can handle 3D or 4D input tensors.

        Args:
            dim: Number of embedding dimensions.
            head_dim: Number of hidden dimensions per head. Default: ``32``
            qkv_bias: Use bias or not. Default: ``False``
            attn_drop: Dropout rate for attention tensor.
            proj_drop: Dropout rate for projection tensor.
        """
        super(MHSA, self).__init__()
        assert dim % head_dim == 0, "dim should be divisible by head_dim"
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.scale = head_dim**-0.5

        self.qkv = nn.Dense(dim, dim * 3, has_bias=qkv_bias)
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.proj = nn.Dense(dim, dim)
        self.proj_drop = nn.Dropout(p=proj_drop)
        self.batch_matmul = ops.BatchMatMul()

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        shape = x.shape
        B, C, H, W = shape
        N = H * W
        if len(shape) == 4:
            x = nn.flatten(x, start_dim=2).transpose((0, -1, -2))  # (B, N, C)
        qkv = (
            self.qkv(x)
            .reshape((B, N, 3, self.num_heads, self.head_dim))
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = ops.Unstack(axis=0)(qkv)

        # trick here to make q@k.t more stable
        attn = self.batch_matmul(q*self.scale, k.transpose(0, 1, -1, -2))
        attn = nn.Softmax(axis=-1)(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose((0, 2, 1, -1)).reshape((B, N, C))
        x = self.proj(x)
        x = self.proj_drop(x)
        if len(shape) == 4:
            x = x.transpose((0, -1, -2)).reshape(B, C, H, W)

        return x


class PatchEmbed(nn.Cell):
    """Convolutional patch embedding layer."""

    def __init__(
        self,
        patch_size: int,
        stride: int,
        in_channels: int,
        embed_dim: int,
        inference_mode: bool = False,
    ) -> None:
        """Build patch embedding layer.

        Args:
            patch_size: Patch size for embedding computation.
            stride: Stride for convolutional embedding layer.
            in_channels: Number of channels of input tensor.
            embed_dim: Number of embedding dimensions.
            inference_mode: Flag to instantiate model in inference mode. Default: ``False``
        """
        super().__init__()
        self.layers = nn.CellList()
        self.layers.append(
            ReparamLargeKernelConv(
                in_channels=in_channels,
                out_channels=embed_dim,
                kernel_size=patch_size,
                stride=stride,
                group=in_channels,
                small_kernel=3,
                inference_mode=inference_mode,
            )
        )
        self.layers.append(
            MobileOneBlock(
                in_channels=embed_dim,
                out_channels=embed_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                group=1,
                inference_mode=inference_mode,
                use_se=False,
                num_conv_branches=1,
            )
        )

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class RepMixer(nn.Cell):
    """Reparameterizable token mixer.

    For more details, please refer to our paper:
    FastViT: A Fast Hybrid Vision Transformer using Structural Reparameterization <https://arxiv.org/pdf/2303.14189.pdf>
    """

    def __init__(
        self,
        dim,
        kernel_size=3,
        use_layer_scale=True,
        layer_scale_init_value=1e-5,
        inference_mode: bool = False,
    ):
        """Build RepMixer Module.

        Args:
            dim: Input feature map dimension. :math:`C_{in}` from an expected input of size :math:`(B, C_{in}, H, W)`.
            kernel_size: Kernel size for spatial mixing. Default: 3
            use_layer_scale: If True, learnable layer scale is used. Default: ``True``
            layer_scale_init_value: Initial value for layer scale. Default: 1e-5
            inference_mode: If True, instantiates model in inference mode. Default: ``False``
        """
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.inference_mode = inference_mode
        self.reparam_conv = None

        if inference_mode:
            self.reparam_conv = nn.Conv2d(
                in_channels=self.dim,
                out_channels=self.dim,
                kernel_size=self.kernel_size,
                stride=1,
                pad_mode='pad',
                padding=self.kernel_size // 2,
                group=self.dim,
                has_bias=True,
            )
        else:
            self.norm = MobileOneBlock(
                dim,
                dim,
                kernel_size,
                padding=kernel_size // 2,
                group=dim,
                use_act=False,
                use_scale_branch=False,
                num_conv_branches=0,
            )
            self.mixer = MobileOneBlock(
                dim,
                dim,
                kernel_size,
                padding=kernel_size // 2,
                group=dim,
                use_act=False,
            )
            self.use_layer_scale = use_layer_scale
            if use_layer_scale:
                self.layer_scale = ms.Parameter(
                    layer_scale_init_value * ops.ones((dim, 1, 1), ms.float32), name='w', requires_grad=True
                )

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        if self.reparam_conv is not None:
            x = self.reparam_conv(x)
            return x
        else:
            if self.use_layer_scale:
                x = x + self.layer_scale * (self.mixer(x) - self.norm(x))
            else:
                x = x + self.mixer(x) - self.norm(x)
            return x

    def reparameterize(self) -> None:
        """Reparameterize mixer and norm into a single
        convolutional layer for efficient inference.
        """
        if self.inference_mode:
            return

        self.mixer.reparameterize()
        self.norm.reparameterize()

        if self.use_layer_scale:
            w = self.mixer.id_tensor + ops.ExpandDims()(self.layer_scale, -1) * (
                self.mixer.reparam_conv.weight - self.norm.reparam_conv.weight
            )
            b = ops.Squeeze()(self.layer_scale) * (
                self.mixer.reparam_conv.bias - self.norm.reparam_conv.bias
            )
        else:
            w = (
                self.mixer.id_tensor
                + self.mixer.reparam_conv.weight
                - self.norm.reparam_conv.weight
            )
            b = self.mixer.reparam_conv.bias - self.norm.reparam_conv.bias

        self.reparam_conv = nn.Conv2d(
            in_channels=self.dim,
            out_channels=self.dim,
            kernel_size=self.kernel_size,
            stride=1,
            pad_mode='pad',
            padding=self.kernel_size // 2,
            group=self.dim,
            has_bias=True,
        )
        self.reparam_conv.weight = w
        self.reparam_conv.bias = b

        for para in self.get_parameters():
            para = ops.stop_gradient(para)
        self.__delattr__("mixer")
        self.__delattr__("norm")
        if self.use_layer_scale:
            self.__delattr__("layer_scale")


class ConvFFN(nn.Cell):
    """Convolutional FFN Module."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        act_layer: nn.Cell = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        """Build convolutional FFN module.

        Args:
            in_channels: Number of input channels.
            hidden_channels: Number of channels after expansion. Default: None
            out_channels: Number of output channels. Default: None
            act_layer: Activation layer. Default: ``GELU``
            drop: Dropout rate. Default: ``0.0``.
        """
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.conv = nn.SequentialCell(
            OrderedDict(
                [("conv", nn.Conv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=7,
                                    pad_mode='pad',
                                    padding=3,
                                    group=in_channels,
                                    has_bias=False,)),
                 ("bn", nn.BatchNorm2d(num_features=out_channels))]))
        self.fc1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        self.drop = nn.Dropout(p=drop)
        self._init_weights()

    def _init_weights(self) -> None:
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(init.initializer(init.TruncatedNormal(sigma=0.02),
                                                      cell.weight.shape,
                                                      cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(init.initializer(init.Zero(), cell.bias.shape, cell.bias.dtype))

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        x = self.conv(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class RepCPE(nn.Cell):
    """Implementation of conditional positional encoding.

    For more details refer to paper:
    `Conditional Positional Encodings for Vision Transformers <https://arxiv.org/pdf/2102.10882.pdf>`_

    In our implementation, we can reparameterize this module to eliminate a skip connection.
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int = 768,
        spatial_shape: Union[int, Tuple[int, int]] = (7, 7),
        inference_mode=False,
    ) -> None:
        """Build reparameterizable conditional positional encoding

        Args:
            in_channels: Number of input channels.
            embed_dim: Number of embedding dimensions. Default: 768
            spatial_shape: Spatial shape of kernel for positional encoding. Default: (7, 7)
            inference_mode: Flag to instantiate block in inference mode. Default: ``False``
        """
        super(RepCPE, self).__init__()
        if isinstance(spatial_shape, int):
            spatial_shape = tuple([spatial_shape] * 2)
        assert isinstance(spatial_shape, Tuple), (
            f'"spatial_shape" must by a sequence or int, '
            f"get {type(spatial_shape)} instead."
        )
        assert len(spatial_shape) == 2, (
            f'Length of "spatial_shape" should be 2, '
            f"got {len(spatial_shape)} instead."
        )

        self.spatial_shape = spatial_shape
        self.embed_dim = embed_dim
        self.in_channels = in_channels
        self.group = embed_dim
        self.reparam_conv = None
        if inference_mode:
            self.reparam_conv = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.embed_dim,
                kernel_size=self.spatial_shape,
                stride=1,
                pad_mode='pad',
                padding=int(self.spatial_shape[0] // 2),
                group=self.embed_dim,
                has_bias=True,
            )
        else:
            self.pe = nn.Conv2d(
                in_channels,
                embed_dim,
                spatial_shape,
                1,
                'pad',
                int(spatial_shape[0] // 2),
                has_bias=True,
                group=embed_dim,
            )

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        if self.reparam_conv is not None:
            x = self.reparam_conv(x)
            return x
        else:
            x = self.pe(x) + x
            return x

    def reparameterize(self) -> None:
        # Build equivalent Id tensor
        input_dim = self.in_channels // self.group
        kernel_value = ops.Zeros()(
            (
                self.in_channels,
                input_dim,
                self.spatial_shape[0],
                self.spatial_shape[1],
            ), ms.float32
        )
        for i in range(self.in_channels):
            kernel_value[
                i,
                i % input_dim,
                self.spatial_shape[0] // 2,
                self.spatial_shape[1] // 2,
            ] = 1
        id_tensor = kernel_value

        # Reparameterize Id tensor and conv
        w_final = id_tensor + self.pe.weight
        b_final = self.pe.bias

        # Introduce reparam conv
        self.reparam_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.embed_dim,
            kernel_size=self.spatial_shape,
            stride=1,
            pad_mode='pad',
            padding=int(self.spatial_shape[0] // 2),
            group=self.embed_dim,
            has_bias=True,
        )
        self.reparam_conv.weight = w_final
        self.reparam_conv.bias = b_final

        for para in self.get_parameters():
            para = ops.stop_gradient(para)
        self.__delattr__("pe")


class RepMixerBlock(nn.Cell):
    """Implementation of Metaformer block with RepMixer as token mixer.

    For more details on Metaformer structure, please refer to:
    `MetaFormer Is Actually What You Need for Vision <https://arxiv.org/pdf/2111.11418.pdf>`_
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 3,
        mlp_ratio: float = 4.0,
        act_layer: nn.Cell = nn.GELU,
        drop: float = 0.0,
        drop_path: float = 0.0,
        use_layer_scale: bool = True,
        layer_scale_init_value: float = 1e-5,
        inference_mode: bool = False,
    ):
        """Build RepMixer Block.

        Args:
            dim: Number of embedding dimensions.
            kernel_size: Kernel size for repmixer. Default: 3
            mlp_ratio: MLP expansion ratio. Default: 4.0
            act_layer: Activation layer. Default: ``nn.GELU``
            drop: Dropout rate. Default: 0.0
            drop_path: Drop path rate. Default: 0.0
            use_layer_scale: Flag to turn on layer scale. Default: ``True``
            layer_scale_init_value: Layer scale value at initialization. Default: 1e-5
            inference_mode: Flag to instantiate block in inference mode. Default: ``False``
        """

        super().__init__()

        self.token_mixer = RepMixer(
            dim,
            kernel_size=kernel_size,
            use_layer_scale=use_layer_scale,
            layer_scale_init_value=layer_scale_init_value,
            inference_mode=inference_mode,
        )

        assert mlp_ratio > 0, "MLP ratio should be greater than 0, found: {}".format(
            mlp_ratio
        )
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.convffn = ConvFFN(
            in_channels=dim,
            hidden_channels=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        # Drop Path
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # Layer Scale
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale = ms.Parameter(
                layer_scale_init_value * ops.ones((dim, 1, 1), ms.float32), requires_grad=True
            )

    def construct(self, x):
        if self.use_layer_scale:
            x = self.token_mixer(x)
            x = x + self.drop_path(self.layer_scale * self.convffn(x))
        else:
            x = self.token_mixer(x)
            x = x + self.drop_path(self.convffn(x))
        return x


class AttentionBlock(nn.Cell):
    """Implementation of metaformer block with MHSA as token mixer.

    For more details on Metaformer structure, please refer to:
    `MetaFormer Is Actually What You Need for Vision <https://arxiv.org/pdf/2111.11418.pdf>`_
    """

    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        act_layer: nn.Cell = nn.GELU,
        norm_layer: nn.Cell = nn.BatchNorm2d,
        drop: float = 0.0,
        drop_path: float = 0.0,
        use_layer_scale: bool = True,
        layer_scale_init_value: float = 1e-5,
    ):
        """Build Attention Block.

        Args:
            dim: Number of embedding dimensions.
            mlp_ratio: MLP expansion ratio. Default: 4.0
            act_layer: Activation layer. Default: ``nn.GELU``
            norm_layer: Normalization layer. Default: ``nn.BatchNorm2d``
            drop: Dropout rate. Default: 0.0
            drop_path: Drop path rate. Default: 0.0
            use_layer_scale: Flag to turn on layer scale. Default: ``True``
            layer_scale_init_value: Layer scale value at initialization. Default: 1e-5
        """

        super().__init__()

        self.norm = norm_layer(dim)
        self.token_mixer = MHSA(dim=dim)

        assert mlp_ratio > 0, "MLP ratio should be greater than 0, found: {}".format(
            mlp_ratio
        )
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.convffn = ConvFFN(
            in_channels=dim,
            hidden_channels=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        # Drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # Layer Scale
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = ms.Parameter(
                layer_scale_init_value * ops.ones((dim, 1, 1), ms.float32), requires_grad=True
            )
            self.layer_scale_2 = ms.Parameter(
                layer_scale_init_value * ops.ones((dim, 1, 1), ms.float32), requires_grad=True
            )

    def construct(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_1 * self.token_mixer(self.norm(x)))
            x = x + self.drop_path(self.layer_scale_2 * self.convffn(x))
        else:
            x = x + self.drop_path(self.token_mixer(self.norm(x)))
            x = x + self.drop_path(self.convffn(x))
        return x


def basic_blocks(
    dim: int,
    block_index: int,
    num_blocks: List[int],
    token_mixer_type: str,
    kernel_size: int = 3,
    mlp_ratio: float = 4.0,
    act_layer: nn.Cell = nn.GELU,
    norm_layer: nn.Cell = nn.BatchNorm2d,
    drop_rate: float = 0.0,
    drop_path_rate: float = 0.0,
    use_layer_scale: bool = True,
    layer_scale_init_value: float = 1e-5,
    inference_mode=False,
) -> nn.SequentialCell:
    """Build FastViT blocks within a stage.

    Args:
        dim: Number of embedding dimensions.
        block_index: block index.
        num_blocks: List containing number of blocks per stage.
        token_mixer_type: Token mixer type.
        kernel_size: Kernel size for repmixer.
        mlp_ratio: MLP expansion ratio.
        act_layer: Activation layer.
        norm_layer: Normalization layer.
        drop_rate: Dropout rate.
        drop_path_rate: Drop path rate.
        use_layer_scale: Flag to turn on layer scale regularization.
        layer_scale_init_value: Layer scale value at initialization.
        inference_mode: Flag to instantiate block in inference mode.

    Returns:
        nn.Sequential object of all the blocks within the stage.
    """
    blocks = []
    for block_idx in range(num_blocks[block_index]):
        block_dpr = (
            drop_path_rate
            * (block_idx + sum(num_blocks[:block_index]))
            / (sum(num_blocks) - 1)
        )
        if token_mixer_type == "repmixer":
            blocks.append(
                RepMixerBlock(
                    dim,
                    kernel_size=kernel_size,
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                    drop=drop_rate,
                    drop_path=block_dpr,
                    use_layer_scale=use_layer_scale,
                    layer_scale_init_value=layer_scale_init_value,
                    inference_mode=inference_mode,
                )
            )
        elif token_mixer_type == "attention":
            blocks.append(
                AttentionBlock(
                    dim,
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    drop=drop_rate,
                    drop_path=block_dpr,
                    use_layer_scale=use_layer_scale,
                    layer_scale_init_value=layer_scale_init_value,
                )
            )
        else:
            raise ValueError(
                "Token mixer type: {} not supported".format(token_mixer_type)
            )
    blocks = nn.SequentialCell(*blocks)

    return blocks


class FastViT(nn.Cell):
    """
    This class implements `FastViT architecture <https://arxiv.org/pdf/2303.14189.pdf>`_
    """

    def __init__(
        self,
        layers,
        token_mixers: Tuple[str, ...],
        embed_dims=None,
        mlp_ratios=None,
        downsamples=None,
        repmixer_kernel_size=3,
        norm_layer: nn.Cell = nn.BatchNorm2d,
        act_layer: nn.Cell = nn.GELU,
        num_classes=1000,
        pos_embs=None,
        down_patch_size=7,
        down_stride=2,
        drop_rate=0.0,
        drop_path_rate=0.0,
        use_layer_scale=True,
        layer_scale_init_value=1e-5,
        fork_feat=False,
        init_cfg=None,
        pretrained=None,
        cls_ratio=2.0,
        inference_mode=False,
        **kwargs,
    ) -> None:

        super().__init__()

        if not fork_feat:
            self.num_classes = num_classes
        self.fork_feat = fork_feat

        if pos_embs is None:
            pos_embs = [None] * len(layers)

        # Convolutional stem
        self.patch_embed = convolutional_stem(3, embed_dims[0], inference_mode)

        # Build the main stages of the network architecture
        self.network = nn.CellList()
        for i in range(len(layers)):
            # Add position embeddings if requested
            if pos_embs[i] is not None:
                self.network.append(
                    pos_embs[i](
                        embed_dims[i], embed_dims[i], inference_mode=inference_mode
                    )
                )
            stage = basic_blocks(
                embed_dims[i],
                i,
                layers,
                token_mixer_type=token_mixers[i],
                kernel_size=repmixer_kernel_size,
                mlp_ratio=mlp_ratios[i],
                act_layer=act_layer,
                norm_layer=norm_layer,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
                inference_mode=inference_mode,
            )
            self.network.append(stage)
            if i >= len(layers) - 1:
                break

            # Patch merging/downsampling between stages.
            if downsamples[i] or embed_dims[i] != embed_dims[i + 1]:
                self.network.append(
                    PatchEmbed(
                        patch_size=down_patch_size,
                        stride=down_stride,
                        in_channels=embed_dims[i],
                        embed_dim=embed_dims[i + 1],
                        inference_mode=inference_mode,
                    )
                )
        # For segmentation and detection, extract intermediate output
        if self.fork_feat:
            # add a norm layer for each output
            self.out_indices = [0, 2, 4, 6]
            for i_emb, i_layer in enumerate(self.out_indices):
                if i_emb == 0 and os.environ.get("FORK_LAST3", None):
                    """For RetinaNet, `start_level=1`. The first norm layer will not used.
                    cmd: `FORK_LAST3=1 python -m torch.distributed.launch ...`
                    """
                    layer = nn.Identity()
                else:
                    layer = norm_layer(embed_dims[i_emb])
                layer_name = f"norm{i_layer}"
                self.insert_child_to_cell(layer_name, layer)
        else:
            # Classifier head
            self.gap = GlobalAvgPooling()
            self.conv_exp = MobileOneBlock(
                in_channels=embed_dims[-1],
                out_channels=int(embed_dims[-1] * cls_ratio),
                kernel_size=3,
                stride=1,
                padding=1,
                group=embed_dims[-1],
                inference_mode=inference_mode,
                use_se=True,
                num_conv_branches=1,
            )
            self.head = (
                nn.Dense(int(embed_dims[-1] * cls_ratio), num_classes)
                if num_classes > 0
                else nn.Identity()
            )

        self.cls_init_weights()
        self.init_cfg = copy.deepcopy(init_cfg)

    def cls_init_weights(self) -> None:
        """Init. for classification"""
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(init.initializer(init.TruncatedNormal(sigma=0.02),
                                                      cell.weight.shape,
                                                      cell.weight.dtype))
            if isinstance(cell, nn.Dense) and cell.bias is not None:
                cell.bias.set_data(init.initializer(init.Zero(), cell.bias.shape, cell.bias.dtype))

    def forward_embeddings(self, x: ms.Tensor) -> ms.Tensor:
        x = self.patch_embed(x)
        return x

    def forward_tokens(self, x: ms.Tensor) -> ms.Tensor:
        outs = []
        for idx, block in enumerate(self.network):
            x = block(x)
            if self.fork_feat and idx in self.out_indices:
                norm_layer = getattr(self, f"norm{idx}")
                x_out = norm_layer(x)
                outs.append(x_out)
        if self.fork_feat:
            # output the features of four stages for dense prediction
            return outs
        # output only the features of last layer for image classification
        return x

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        # input embedding
        x = self.forward_embeddings(x)
        # through backbone
        x = self.forward_tokens(x)
        if self.fork_feat:
            # output features of four stages for dense prediction
            return x
        # for image classification
        x = self.conv_exp(x)
        x = self.gap(x)
        x = x.view((x.shape[0], -1))
        cls_out = self.head(x)
        return cls_out


@register_model
def fastvit_t8(pretrained=False, **kwargs):
    """Instantiate FastViT-T8 model variant."""
    layers = [2, 2, 4, 2]
    embed_dims = [48, 96, 192, 384]
    mlp_ratios = [3, 3, 3, 3]
    downsamples = [True, True, True, True]
    token_mixers = ("repmixer", "repmixer", "repmixer", "repmixer")
    model = FastViT(
        layers,
        token_mixers=token_mixers,
        embed_dims=embed_dims,
        mlp_ratios=mlp_ratios,
        downsamples=downsamples,
        **kwargs,
    )
    model.default_cfg = default_cfgs["fastvit_t"]
    if pretrained:
        raise ValueError("Functionality not implemented.")
    return model


@register_model
def fastvit_t12(pretrained=False, **kwargs):
    """Instantiate FastViT-T12 model variant."""
    layers = [2, 2, 6, 2]
    embed_dims = [64, 128, 256, 512]
    mlp_ratios = [3, 3, 3, 3]
    downsamples = [True, True, True, True]
    token_mixers = ("repmixer", "repmixer", "repmixer", "repmixer")
    model = FastViT(
        layers,
        token_mixers=token_mixers,
        embed_dims=embed_dims,
        mlp_ratios=mlp_ratios,
        downsamples=downsamples,
        **kwargs,
    )
    model.default_cfg = default_cfgs["fastvit_t"]
    if pretrained:
        raise ValueError("Functionality not implemented.")
    return model


@register_model
def fastvit_s12(pretrained=False, **kwargs):
    """Instantiate FastViT-S12 model variant."""
    layers = [2, 2, 6, 2]
    embed_dims = [64, 128, 256, 512]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    token_mixers = ("repmixer", "repmixer", "repmixer", "repmixer")
    model = FastViT(
        layers,
        token_mixers=token_mixers,
        embed_dims=embed_dims,
        mlp_ratios=mlp_ratios,
        downsamples=downsamples,
        **kwargs,
    )
    model.default_cfg = default_cfgs["fastvit_s"]
    if pretrained:
        raise ValueError("Functionality not implemented.")
    return model


@register_model
def fastvit_sa12(pretrained=False, **kwargs):
    """Instantiate FastViT-SA12 model variant."""
    layers = [2, 2, 6, 2]
    embed_dims = [64, 128, 256, 512]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    pos_embs = [None, None, None, partial(RepCPE, spatial_shape=(7, 7))]
    token_mixers = ("repmixer", "repmixer", "repmixer", "attention")
    model = FastViT(
        layers,
        token_mixers=token_mixers,
        embed_dims=embed_dims,
        pos_embs=pos_embs,
        mlp_ratios=mlp_ratios,
        downsamples=downsamples,
        **kwargs,
    )
    model.default_cfg = default_cfgs["fastvit_s"]
    if pretrained:
        raise ValueError("Functionality not implemented.")
    return model


@register_model
def fastvit_sa24(pretrained=False, **kwargs):
    """Instantiate FastViT-SA24 model variant."""
    layers = [4, 4, 12, 4]
    embed_dims = [64, 128, 256, 512]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    pos_embs = [None, None, None, partial(RepCPE, spatial_shape=(7, 7))]
    token_mixers = ("repmixer", "repmixer", "repmixer", "attention")
    model = FastViT(
        layers,
        token_mixers=token_mixers,
        embed_dims=embed_dims,
        pos_embs=pos_embs,
        mlp_ratios=mlp_ratios,
        downsamples=downsamples,
        **kwargs,
    )
    model.default_cfg = default_cfgs["fastvit_s"]
    if pretrained:
        raise ValueError("Functionality not implemented.")
    return model


@register_model
def fastvit_sa36(pretrained=False, **kwargs):
    """Instantiate FastViT-SA36 model variant."""
    layers = [6, 6, 18, 6]
    embed_dims = [64, 128, 256, 512]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    pos_embs = [None, None, None, partial(RepCPE, spatial_shape=(7, 7))]
    token_mixers = ("repmixer", "repmixer", "repmixer", "attention")
    model = FastViT(
        layers,
        embed_dims=embed_dims,
        token_mixers=token_mixers,
        pos_embs=pos_embs,
        mlp_ratios=mlp_ratios,
        downsamples=downsamples,
        layer_scale_init_value=1e-6,
        **kwargs,
    )
    model.default_cfg = default_cfgs["fastvit_m"]
    if pretrained:
        raise ValueError("Functionality not implemented.")
    return model


@register_model
def fastvit_ma36(pretrained=False, **kwargs):
    """Instantiate FastViT-MA36 model variant."""
    layers = [6, 6, 18, 6]
    embed_dims = [76, 152, 304, 608]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    pos_embs = [None, None, None, partial(RepCPE, spatial_shape=(7, 7))]
    token_mixers = ("repmixer", "repmixer", "repmixer", "attention")
    model = FastViT(
        layers,
        embed_dims=embed_dims,
        token_mixers=token_mixers,
        pos_embs=pos_embs,
        mlp_ratios=mlp_ratios,
        downsamples=downsamples,
        layer_scale_init_value=1e-6,
        **kwargs,
    )
    model.default_cfg = default_cfgs["fastvit_m"]
    if pretrained:
        raise ValueError("Functionality not implemented.")
    return model


class DropPath(nn.Cell):
    """DropPath (Stochastic Depth) regularization layers"""

    def __init__(
        self,
        drop_prob: float = 0.0,
        scale_by_keep: bool = True,
    ) -> None:
        super().__init__()
        self.keep_prob = 1.0 - drop_prob
        self.scale_by_keep = scale_by_keep
        self.dropout = nn.Dropout(p=drop_prob)

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        if self.keep_prob == 1.0 or not self.training:
            return x
        shape = (x.shape[0], ) + (1,) * (x.ndim - 1)
        random_tensor = self.dropout(ones(shape))
        if not self.scale_by_keep:
            random_tensor = ops.mul(random_tensor, self.keep_prob)
        return x * random_tensor


class SEBlock(nn.Cell):

    def __init__(self, in_channels: int, rd_ratio: float = 0.0625) -> None:
        """Construct a Squeeze and Excite Module.

        Args:
            in_channels: Number of input channels.
            rd_ratio: Input channel reduction ratio.
        """
        super(SEBlock, self).__init__()
        self.reduce = nn.Conv2d(
            in_channels=in_channels,
            out_channels=int(in_channels * rd_ratio),
            kernel_size=1,
            stride=1,
            pad_mode='valid',
            has_bias=True,
        )
        self.expand = nn.Conv2d(
            in_channels=int(in_channels * rd_ratio),
            out_channels=in_channels,
            kernel_size=1,
            pad_mode='valid',
            stride=1,
            has_bias=True,
        )

    def construct(self, inputs: ms.Tensor) -> ms.Tensor:
        """Apply forward pass."""
        b, c, h, w = inputs.shape
        x = ops.AvgPool(pad_mode='valid', kernel_size=(h, w))(inputs)
        x = self.reduce(x)
        x = nn.ReLU()(x)
        x = self.expand(x)
        x = nn.Sigmoid()(x)
        x = x.view((-1, c, 1, 1))
        return inputs * x


class MobileOneBlock(nn.Cell):
    """MobileOne building block.

    This block has a multi-branched architecture at train-time
    and plain-CNN style architecture at inference time
    For more details, please refer to our paper:
    `An Improved One millisecond Mobile Backbone` -
    https://arxiv.org/pdf/2206.04040.pdf
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        group: int = 1,
        inference_mode: bool = False,
        use_se: bool = False,
        use_act: bool = True,
        use_scale_branch: bool = True,
        num_conv_branches: int = 1,
        activation: nn.Cell = nn.GELU,
    ) -> None:
        """Construct a MobileOneBlock module.

        Args:
            in_channels: Number of channels in the input.
            out_channels: Number of channels produced by the block.
            kernel_size: Size of the convolution kernel.
            stride: Stride size.
            padding: Zero-padding size.
            dilation: Kernel dilation factor.
            group: Group number.
            inference_mode: If True, instantiates model in inference mode.
            use_se: Whether to use SE-ReLU activations.
            use_act: Whether to use activation. Default: ``True``
            use_scale_branch: Whether to use scale branch. Default: ``True``
            num_conv_branches: Number of linear conv branches.
        """
        super(MobileOneBlock, self).__init__()
        self.inference_mode = inference_mode
        self.group = group
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_conv_branches = num_conv_branches

        # Check if SE-ReLU is requested
        if use_se:
            self.se = SEBlock(out_channels)
        else:
            self.se = nn.Identity()

        if use_act:
            self.activation = activation()
        else:
            self.activation = nn.Identity()

        if inference_mode:
            self.reparam_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                pad_mode='pad',
                padding=padding,
                dilation=dilation,
                group=group,
                has_bias=True,
            )
        else:
            # Re-parameterizable skip connection
            self.rbr_skip = (
                nn.BatchNorm2d(num_features=in_channels)
                if out_channels == in_channels and stride == 1
                else None
            )

            # Re-parameterizable conv branches
            if num_conv_branches > 0:
                self.rbr_conv = nn.CellList()
                for _ in range(self.num_conv_branches):
                    self.rbr_conv.append(
                        self._conv_bn(kernel_size=kernel_size, padding=padding)
                    )
            else:
                self.rbr_conv = None

            # Re-parameterizable scale branch
            self.rbr_scale = None
            if (kernel_size > 1) and use_scale_branch:
                self.rbr_scale = self._conv_bn(kernel_size=1, padding=0)

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        """Apply forward pass."""
        # Inference mode forward pass.
        if self.inference_mode:
            return self.activation(self.se(self.reparam_conv(x)))

        # Multi-branched train-time forward pass.
        # Skip branch output
        identity_out = 0
        if self.rbr_skip is not None:
            identity_out = self.rbr_skip(x)

        # Scale branch output
        scale_out = 0
        if self.rbr_scale is not None:
            scale_out = self.rbr_scale(x)

        # Other branches
        out = scale_out + identity_out
        if self.rbr_conv is not None:
            for ix in range(self.num_conv_branches):
                out += self.rbr_conv[ix](x)

        return self.activation(self.se(out))

    def reparameterize(self):
        """Following works like `RepVGG: Making VGG-style ConvNets Great Again` -
        https://arxiv.org/pdf/2101.03697.pdf. We re-parameterize multi-branched
        architecture used at training time to obtain a plain CNN-like structure
        for inference.
        """
        if self.inference_mode:
            return
        kernel, bias = self._get_kernel_bias()
        self.reparam_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            pad_mode='pad',
            padding=self.padding,
            dilation=self.dilation,
            group=self.group,
            has_bias=True,
        )
        self.reparam_conv.weight = kernel
        self.reparam_conv.bias = bias

        # Delete un-used branches
        for para in self.get_parameters():
            para = ops.stop_gradient(para)
        self.__delattr__("rbr_conv")
        self.__delattr__("rbr_scale")
        if hasattr(self, "rbr_skip"):
            self.__delattr__("rbr_skip")

        self.inference_mode = True

    def _get_kernel_bias(self) -> Tuple[ms.Tensor, ms.Tensor]:
        """Method to obtain re-parameterized kernel and bias.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L83

        Returns:
            Tuple of (kernel, bias) after fusing branches.
        """
        # get weights and bias of scale branch
        kernel_scale = 0
        bias_scale = 0
        if self.rbr_scale is not None:
            kernel_scale, bias_scale = self._fuse_bn_tensor(self.rbr_scale)
            # Pad scale branch kernel to match conv branch kernel size.
            pad = self.kernel_size // 2
            pad_op = nn.Pad(paddings=((0, 0), (0, 0), (pad, pad), (pad, pad)))
            kernel_scale = pad_op(kernel_scale)

        # get weights and bias of skip branch
        kernel_identity = 0
        bias_identity = 0
        if self.rbr_skip is not None:
            kernel_identity, bias_identity = self._fuse_bn_tensor(self.rbr_skip)

        # get weights and bias of conv branches
        kernel_conv = 0
        bias_conv = 0
        if self.rbr_conv is not None:
            for ix in range(self.num_conv_branches):
                _kernel, _bias = self._fuse_bn_tensor(self.rbr_conv[ix])
                kernel_conv += _kernel
                bias_conv += _bias

        kernel_final = kernel_conv + kernel_scale + kernel_identity
        bias_final = bias_conv + bias_scale + bias_identity
        return kernel_final, bias_final

    def _fuse_bn_tensor(
        self, branch: Union[nn.SequentialCell, nn.BatchNorm2d]
    ) -> Tuple[ms.Tensor, ms.Tensor]:
        """Method to fuse batchnorm layer with preceeding conv layer.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L95

        Args:
            branch: Sequence of ops to be fused.

        Returns:
            Tuple of (kernel, bias) after fusing batchnorm.
        """
        if isinstance(branch, nn.SequentialCell):
            kernel = branch.conv.weight
            running_mean = branch.bn.moving_mean
            running_var = branch.bn.moving_variance
            gamma = branch.bn.gamma
            beta = branch.bn.beta
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.group
                kernel_value = ops.zeros(
                    (self.in_channels, input_dim, self.kernel_size, self.kernel_size),
                    ms.float32
                )
                for i in range(self.in_channels):
                    kernel_value[
                        i, i % input_dim, self.kernel_size // 2, self.kernel_size // 2
                    ] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch.moving_mean
            running_var = branch.moving_variance
            gamma = branch.gamma
            beta = branch.beta
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def _conv_bn(self, kernel_size: int, padding: int) -> nn.SequentialCell:
        """Helper method to construct conv-batchnorm layers.

        Args:
            kernel_size: Size of the convolution kernel.
            padding: Zero-padding size.

        Returns:
            Conv-BN module.
        """
        mod_list = nn.SequentialCell(
            OrderedDict(
                [("conv", nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=kernel_size,
                    stride=self.stride,
                    pad_mode='pad',
                    padding=padding,
                    group=self.group,
                    has_bias=False, )),
                 ("bn", nn.BatchNorm2d(num_features=self.out_channels))]))
        return mod_list


class ReparamLargeKernelConv(nn.Cell):
    """Building Block of RepLKNet

    This class defines overparameterized large kernel conv block
    introduced in `RepLKNet <https://arxiv.org/abs/2203.06717>`_

    Reference: https://github.com/DingXiaoH/RepLKNet-pytorch
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        group: int,
        small_kernel: int,
        inference_mode: bool = False,
        activation: nn.Cell = nn.GELU,
    ) -> None:
        """Construct a ReparamLargeKernelConv module.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Kernel size of the large kernel conv branch.
            stride: Stride size. Default: 1
            groups: Group number. Default: 1
            small_kernel: Kernel size of small kernel conv branch.
            inference_mode: If True, instantiates model in inference mode. Default: ``False``
            activation: Activation module. Default: ``nn.GELU``
        """
        super(ReparamLargeKernelConv, self).__init__()

        self.stride = stride
        self.group = group
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation()

        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        self.padding = kernel_size // 2
        self.lkb_reparam = None
        self.small_conv = None
        if inference_mode:
            self.lkb_reparam = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                pad_mode='pad',
                padding=self.padding,
                dilation=1,
                group=group,
                has_bias=True,
            )
        else:
            self.lkb_origin = self._conv_bn(
                kernel_size=kernel_size, padding=self.padding
            )
            if small_kernel is not None:
                assert (
                    small_kernel <= kernel_size
                ), "The kernel size for re-param cannot be larger than the large kernel!"
                self.small_conv = self._conv_bn(
                    kernel_size=small_kernel, padding=small_kernel // 2
                )

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        """Apply forward pass."""
        if self.lkb_reparam is not None:
            out = self.lkb_reparam(x)
        else:
            out = self.lkb_origin(x)
            if self.small_conv is not None:
                out += self.small_conv(x)

        self.activation(out)
        return out

    def get_kernel_bias(self) -> Tuple[ms.Tensor, ms.Tensor]:
        """Method to obtain re-parameterized kernel and bias.
        Reference: https://github.com/DingXiaoH/RepLKNet-pytorch

        Returns:
            Tuple of (kernel, bias) after fusing branches.
        """
        eq_k, eq_b = self._fuse_bn(self.lkb_origin.conv, self.lkb_origin.bn)
        if hasattr(self, "small_conv"):
            small_k, small_b = self._fuse_bn(self.small_conv.conv, self.small_conv.bn)
            eq_b += small_b
            pad_op = nn.Pad(paddings=((0, 0), (0, 0), ((self.kernel_size - self.small_kernel) // 2,
                                                       (self.kernel_size - self.small_kernel) // 2),
                                                      ((self.kernel_size - self.small_kernel) // 2,
                                                       (self.kernel_size - self.small_kernel) // 2)))
            eq_k += pad_op(small_k)
        return eq_k, eq_b

    def reparameterize(self) -> None:
        """
        Following works like `RepVGG: Making VGG-style ConvNets Great Again` -
        https://arxiv.org/pdf/2101.03697.pdf. We re-parameterize multi-branched
        architecture used at training time to obtain a plain CNN-like structure
        for inference.
        """
        eq_k, eq_b = self.get_kernel_bias()
        self.lkb_reparam = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            pad_mode='pad',
            padding=self.padding,
            dilation=self.lkb_origin.conv.dilation,
            group=self.group,
            has_bias=True,
        )

        self.lkb_reparam.weight = eq_k
        self.lkb_reparam.bias = eq_b
        self.__delattr__("lkb_origin")
        if hasattr(self, "small_conv"):
            self.__delattr__("small_conv")

    @staticmethod
    def _fuse_bn(
        conv: ms.Tensor, bn: nn.BatchNorm2d
    ) -> Tuple[ms.Tensor, ms.Tensor]:
        """Method to fuse batchnorm layer with conv layer.

        Args:
            conv: Convolutional kernel weights.
            bn: Batchnorm 2d layer.

        Returns:
            Tuple of (kernel, bias) after fusing batchnorm.
        """
        kernel = conv.weight
        running_mean = bn.moving_mean
        running_var = bn.moving_variance
        gamma = bn.gamma
        beta = bn.beta
        eps = bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def _conv_bn(self, kernel_size: int, padding: int = 0) -> nn.SequentialCell:
        """Helper method to construct conv-batchnorm layers.

        Args:
            kernel_size: Size of the convolution kernel.
            padding: Zero-padding size.

        Returns:
            A nn.Sequential Conv-BN module.
        """
        mod_list = nn.SequentialCell(
            OrderedDict(
                [("conv", nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=kernel_size,
                    stride=self.stride,
                    pad_mode='pad',
                    padding=padding,
                    group=self.group,
                    has_bias=False,)),
                 ("bn", nn.BatchNorm2d(num_features=self.out_channels))]))
        return mod_list


def reparameterize_model(model: nn.Cell) -> nn.Cell:
    """Method returns a model where a multi-branched structure
        used in training is re-parameterized into a single branch
        for inference.

    Args:
        model: MobileOne model in train mode.

    Returns:
        MobileOne model in inference mode.
    """
    # Avoid editing original graph
    model = copy.deepcopy(model)
    for _, cell in model.cells_and_names():
        if hasattr(cell, "reparameterize"):
            cell.reparameterize()
    return model


class CosineWDSchedule:
    def __init__(self, optimizer, t_max, eta_min=0, last_epoch=-1):
        self.last_epoch = last_epoch
        self.base_wds = [group["weight_decay"] for group in optimizer.param_groups]
        self.t_max = t_max
        self.eta_min = eta_min

    def _get_wd(self, optimizer):
        if self.last_epoch == 0:
            return self.base_wds
        elif (self.last_epoch - 1 - self.t_max) % (2 * self.t_max) == 0:
            return [
                group["weight_decay"]
                + (base_lr - self.eta_min) * (1 - math.cos(math.pi / self.t_max)) / 2
                for base_lr, group in zip(self.base_wds, optimizer.param_groups)
            ]
        return [
            (1 + math.cos(math.pi * self.last_epoch / self.t_max))
            / (1 + math.cos(math.pi * (self.last_epoch - 1) / self.t_max))
            * (group["weight_decay"] - self.eta_min)
            + self.eta_min
            for group in optimizer.param_groups
        ]

    def update_weight_decay(self, optimizer):
        self.last_epoch += 1
        values = self._get_wd(optimizer)
        for i, data in enumerate(zip(optimizer.param_groups, values)):
            param_group, wd = data
            # Avoid updating weight decay of param_groups that should not be decayed.
            if param_group["weight_decay"] > 0.0:
                param_group["weight_decay"] = wd


class DistillationLoss(nn.Cell):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """

    def __init__(
        self,
        base_criterion: nn.Cell,
        teacher_model: nn.Cell,
        distillation_type: str,
        alpha: float,
        tau: float,
    ):
        super(DistillationLoss, self).__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        assert distillation_type in ["none", "soft", "hard"]
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau

    def construct(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model.
            outputs: Output tensor from model being trained.
            labels: the labels for the base criterion.
        """
        base_loss = self.base_criterion(outputs, labels)
        if self.distillation_type == "none":
            return base_loss
        teacher_outputs = self.teacher_model(inputs)
        teacher_outputs = ops.stop_gradient(teacher_outputs)
        if self.distillation_type == "soft":
            T = self.tau
            distillation_loss = (
                ops.KLDivLoss(
                    nn.LogSoftmax(outputs / T, axis=1),
                    nn.LogSoftmax(teacher_outputs / T, axis=1),
                    reduction="sum",
                )
                * (T * T)
                / ops.Size()(outputs)
            )
        elif self.distillation_type == "hard":
            distillation_loss = ops.cross_entropy(outputs, ops.Argmax(axis=1)(teacher_outputs))

        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
        return loss
