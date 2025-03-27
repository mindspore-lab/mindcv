""" Flatten Module"""
from mindspore import Tensor, nn


class Flatten(nn.Cell):
    """
    Flattens a contiguous range of dims into a tensor.
    """
    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def construct(self, input: Tensor) -> Tensor:
        return input.flatten(start_dim=self.start_dim, end_dim=self.end_dim)
