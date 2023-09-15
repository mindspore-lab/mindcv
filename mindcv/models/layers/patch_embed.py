""" Image to Patch Embedding using Conv2d
A convolution based approach to patchifying a 2D image w/ embedding projection."""
from typing import Optional

from mindspore import Tensor, nn, ops

from .format import Format, nchw_to
from .helpers import to_2tuple


class PatchEmbed(nn.Cell):
    """Image to Patch Embedding

    Args:
        image_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Cell, optional): Normalization layer. Default: None
    """
    output_fmt: Format

    def __init__(
        self,
        image_size: Optional[int] = 224,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 96,
        norm_layer: Optional[nn.Cell] = None,
        flatten: bool = True,
        output_fmt: Optional[str] = None,
        bias: bool = True,
        strict_img_size: bool = True,
        dynamic_img_pad: bool = False,
    ) -> None:
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        if image_size is not None:
            self.image_size = to_2tuple(image_size)
            self.patches_resolution = tuple([s // p for s, p in zip(self.image_size, self.patch_size)])
            self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        else:
            self.image_size = None
            self.patches_resolution = None
            self.num_patches = None

        if output_fmt is not None:
            self.flatten = False
            self.output_fmt = Format(output_fmt)
        else:
            self.flatten = flatten
            self.output_fmt = Format.NCHW

        self.strict_img_size = strict_img_size
        self.dynamic_img_pad = dynamic_img_pad
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_channels=in_chans, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size,
                              pad_mode='pad', has_bias=bias, weight_init="TruncatedNormal")

        if norm_layer is not None:
            if isinstance(embed_dim, int):
                embed_dim = (embed_dim,)
            self.norm = norm_layer(embed_dim, epsilon=1e-5)
        else:
            self.norm = None

    def construct(self, x: Tensor) -> Tensor:
        """docstring"""
        B, C, H, W = x.shape
        if self.image_size is not None:
            if self.strict_img_size:
                if (H, W) != (self.image_size[0], self.image_size[1]):
                    raise ValueError(f"Input height and width ({H},{W}) doesn't match model ({self.image_size[0]},"
                                     f"{self.image_size[1]}).")
            elif not self.dynamic_img_pad:
                if H % self.patch_size[0] != 0:
                    raise ValueError(f"Input height ({H}) should be divisible by patch size ({self.patch_size[0]}).")
                if W % self.patch_size[1] != 0:
                    raise ValueError(f"Input width ({W}) should be divisible by patch size ({self.patch_size[1]}).")
        if self.dynamic_img_pad:
            pad_h = (self.patch_size[0] - H % self.patch_size[0]) % self.patch_size[0]
            pad_w = (self.patch_size[1] - W % self.patch_size[1]) % self.patch_size[1]
            x = ops.pad(x, (0, pad_w, 0, pad_h))

        # FIXME look at relaxing size constraints
        x = self.proj(x)
        if self.flatten:
            x = ops.Reshape()(x, (B, self.embed_dim, -1))  # B Ph*Pw C
            x = ops.Transpose()(x, (0, 2, 1))
        elif self.output_fmt != "NCHW":
            x = nchw_to(x, self.output_fmt)
        if self.norm is not None:
            x = self.norm(x)
        return x
