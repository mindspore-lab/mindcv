"""DropPath
Mindspore implementations of DropPath (Stochastic Depth) regularization layers.
Papers:
Deep Networks with Stochastic Depth (https://arxiv.org/abs/1603.09382)
"""
from mindspore import nn
from mindspore import ops
from mindspore import Tensor
from mindspore.numpy import empty


def drop_path(x: Tensor,
              drop_prob: float = 0.,
              training: bool = False,
              scale_by_keep: bool = True) -> Tensor:
    """ DropPath (Stochastic Depth) regularization layers """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = ops.bernoulli(empty(shape), p=keep_prob)
    if keep_prob > 0. and scale_by_keep:
        random_tensor = ops.div(random_tensor, keep_prob)
    return x * random_tensor


class DropPath(nn.Cell):
    """ DropPath (Stochastic Depth) regularization layers """
    def __init__(self,
                 drop_prob: float = 0.,
                 scale_by_keep: bool = True) -> None:
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def construct(self, x: Tensor) -> Tensor:
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)
