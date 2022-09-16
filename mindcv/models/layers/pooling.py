import mindspore.nn as nn
import mindspore.ops as ops


class GlobalAvgPooling(nn.Cell):

    def __init__(self,
                 keep_dims: bool = False
                 ) -> None:
        super(GlobalAvgPooling, self).__init__()
        self.keep_dims = keep_dims

    def construct(self, x):
        x = ops.mean(x, axis=(2, 3), keep_dims=self.keep_dims)
        return x
