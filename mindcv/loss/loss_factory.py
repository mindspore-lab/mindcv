''' loss factory '''
from typing import Optional
from mindspore import Tensor
from .cross_entropy_smooth import CrossEntropySmooth
from .binary_cross_entropy_smooth import BinaryCrossEntropySmooth

__all__ = ["create_loss"]


def create_loss(
        name: str = 'CE',
        weight: Optional[Tensor] = None,
        reduction: str = 'mean',
        label_smoothing: float = 0.,
        aux_factor: float = 0.):
    r"""Creates loss function

    Args:
        name (str):  loss name, : 'CE' for cross_entropy. 'BCE': binary cross entropy. Default: 'CE'.
        weight (Tensor): Class weight. Shape [C]. A rescaling weight applied to the loss of each batch element.
                Data type must be float16 or float32.
        reduction: Apply specific reduction method to the output: 'mean' or 'sum'. Default: 'mean'.
        label_smoothing: Label smoothing factor, a regularization tool used to prevent the model
            from overfitting when calculating Loss. The value range is [0.0, 1.0]. Default: 0.0.
        aux_factor (float): Auxiliary loss factor. Set aux_fuactor > 0.0 if the model has auxilary logit outputs (i.e., deep supervision), like inception_v3.  Default: 0.0.

    Inputs:
        - logits (Tensor or Tuple of Tensor): Input logits. Shape [N, C], where N is # samples, C is # classes.
                Tuple of two input logits are supported in order (main_logits, aux_logits) for auxilary loss used in networks like inception_v3.
          where `C = number of classes`. Data type must be float16 or float32.
        - labels (Tensor): Ground truth labels. Shape: [N] or [N, C].
                (1) shape (N), sparse labels representing the class indinces. Must be int type,
                (2) shape [N, C], dense labels representing the ground truth class probability values, or the one-hot labels. Must be float type.
                If the loss type is BCE, the shape of labels must be [N, C].
    Returns:
       Loss function to compute the loss between the input logits and labels.

    """
    name = name.lower()

    if name == 'ce':
        loss = CrossEntropySmooth(smoothing=label_smoothing, aux_factor=aux_factor, reduction=reduction, weight=weight)
    elif name == 'bce':
        loss = BinaryCrossEntropySmooth(smoothing=label_smoothing, aux_factor=aux_factor, reduction=reduction, weight=weight, pos_weight=None)
    else:
        raise NotImplementedError

    return loss
