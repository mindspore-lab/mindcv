"""
Custom operators.
"""

from mindspore import nn
from mindspore import ops

__all__ = ['Swish']


class Swish(nn.Cell):
    """
    Swish activation function: x * sigmoid(x).

    Args:
        None

    Return:
        Tensor

    Example:
        >>> x = Tensor(((20, 16), (50, 50)), mindspore.float32)
        >>> Swish()(x)
    """

    def __init__(self):
        super(Swish, self).__init__()
        self.result = None
        self.sigmoid = ops.Sigmoid()

    def construct(self, x):
        """ construct swish """
        result = x * self.sigmoid(x)
        return result

    def bprop(self, x, dout):
        """ bprop """
        sigmoid_x = self.sigmoid(x)
        result = dout * (sigmoid_x * (1 + x * (1 - sigmoid_x)))
        return result
