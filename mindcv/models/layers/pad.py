""" Pad Module"""
import mindspore.mint.nn.functional as F
from mindspore import nn


class Pad(nn.Cell):
    """
    Pad Module to pad the input tensor according to the paddings and mode.
    """
    def __init__(self, pad, mode='constant', value=0.0) -> None:
        super().__init__()
        self.pad = pad
        self.mode = mode
        self.value = value

    def construct(self, x):
        x = F.pad(x, self.pad, self.mode, self.value)
        return x
