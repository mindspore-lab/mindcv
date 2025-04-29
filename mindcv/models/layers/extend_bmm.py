""" Extended Batch MatMul Module"""
from mindspore import mint, nn


class ExtendBatchMatMul(nn.Cell):
    """
    Extend Batch MatMul Module to deal with batch matrix multiplication between tensors with higher dimensions
    """

    def __init__(self, transpose_a=False, transpose_b=False) -> None:
        super().__init__()
        self.transpose_a = transpose_a
        self.transpose_b = transpose_b

    def construct(self, a, b):
        if self.transpose_a:
            a = mint.transpose(a, -1, -2)
        if self.transpose_b:
            b = mint.transpose(b, -1, -2)
        size = len(a.shape)
        if size <= 3:
            return mint.bmm(a, b)
        output_shape = (*a.shape[:-2], a.shape[-2], b.shape[-1])
        a = mint.reshape(a, (-1, *a.shape[-2:]))
        b = mint.reshape(b, (-1, *b.shape[-2:]))

        return mint.reshape(mint.bmm(a, b), output_shape)
