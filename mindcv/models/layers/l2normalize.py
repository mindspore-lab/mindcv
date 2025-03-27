""" L2Normalize Module"""
from mindspore import mint


class L2Normalize:
    def __init__(self, axis=-1, epsilon=1e-12):
        """
        Initializes the L2Normalize class in PyTorch.

        :param axis: Specifies the axis along which normalization is applied, default is -1 (last axis).
        :param epsilon: A small value added to the norm to avoid division by zero, default is 1e-12.
        """
        self.axis = axis
        self.epsilon = epsilon

    def __call__(self, input_tensor):
        """
        Applies L2 normalization to the input tensor.

        :param input_tensor: The input tensor to be normalized.
        :return: The L2 normalized tensor.
        """
        norm = mint.sqrt(mint.sum(input_tensor ** 2, dim=self.axis, keepdim=True) + self.epsilon)
        output_tensor = input_tensor / norm
        return output_tensor
