"""Reduces the tensor data across all devices in such a way that all devices will get the same final result."""
from mindspore import nn, ops
from mindspore.ops import ReduceOp

class Allreduce(nn.Cell):
    """Reduces the tensor data across all devices in such a way that all devices will get the same final result."""
    def __init__(self):
        super(Allreduce, self).__init__()
        self.allreduce_sum = ops.AllReduce(ReduceOp.SUM)

    def construct(self, x):
        return self.allreduce_sum(x)
