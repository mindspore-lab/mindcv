""" GlobalAvgPooling Module"""
from mindspore import mint, nn


class GlobalAvgPooling(nn.Cell):
    """
    GlobalAvgPooling, same as torch.nn.AdaptiveAvgPool2d when output shape is 1
    """

    def __init__(self, keep_dims: bool = False) -> None:
        super().__init__()
        self.keep_dims = keep_dims

    def construct(self, x):
        x = mint.mean(x, dim=(2, 3), keepdim=self.keep_dims)
        return x
