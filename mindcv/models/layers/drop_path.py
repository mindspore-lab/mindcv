import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.numpy import empty


def drop_path(x: Tensor,
              drop_prob: float = 0.,
              training: bool = False,
              scale_by_keep: bool = True) -> Tensor:
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = ops.bernoulli(empty(shape), p=keep_prob)
    if keep_prob > 0. and scale_by_keep:
        random_tensor = ops.div(random_tensor, keep_prob)
    return x * random_tensor


class DropPath(nn.Cell):

    def __init__(self,
                 drop_prob: float = 0.,
                 scale_by_keep: bool = True) -> None:
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def construct(self, x: Tensor) -> Tensor:
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)
