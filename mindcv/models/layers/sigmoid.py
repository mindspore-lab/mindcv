""" Sigmoid Module"""
from mindspore import mint, nn


class Sigmoid(nn.Cell):
    def construct(self, x):
        x = mint.sigmoid(x)
        return x
