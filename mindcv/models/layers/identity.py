import mindspore.nn as nn


class Identity(nn.Cell):
    """Identity"""

    def construct(self, x):
        return x
