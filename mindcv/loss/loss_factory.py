""" loss factory """
from typing import Optional

from mindspore import Tensor

from .asymmetric import AsymmetricLossMultilabel, AsymmetricLossSingleLabel
from .binary_cross_entropy_smooth import BinaryCrossEntropySmooth
from .cross_entropy_smooth import CrossEntropySmooth
from .jsd import JSDCrossEntropy

__all__ = ["create_loss"]


def create_loss(
    name: str = "CE",
    weight: Optional[Tensor] = None,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
    aux_factor: float = 0.0,
):
    r"""Creates loss function

    Args:
        name (str):  loss name : 'CE' for cross_entropy. 'BCE': binary cross entropy. Default: 'CE'.
        weight (Tensor): Class weight. A rescaling weight given to the loss of each batch element.
            If given, has to be a Tensor of size 'nbatch'. Data type must be float16 or float32.
        reduction: Apply specific reduction method to the output: 'mean' or 'sum'.
            By default, the sum of the output will be divided by the number of elements in the output.
            'sum': the output will be summed. Default:'mean'.
        label_smoothing: Label smoothing factor, a regularization tool used to prevent the model
            from overfitting when calculating Loss. The value range is [0.0, 1.0]. Default: 0.0.
        aux_factor (float): Auxiliary loss factor. Set aux_factor > 0.0 if the model has auxiliary logit outputs
            (i.e., deep supervision), like inception_v3. Default: 0.0.

    Inputs:
        - logits (Tensor or Tuple of Tensor): Input logits. Shape [N, C], where N means the number of samples,
            C means number of classes. Tuple of two input logits are supported in order (main_logits, aux_logits)
            for auxiliary loss used in networks like inception_v3. Data type must be float16 or float32.
        - labels (Tensor): Ground truth labels. Shape: [N] or [N, C].
            (1) If in shape [N], sparse labels representing the class indices. Must be int type.
            (2) shape [N, C], dense labels representing the ground truth class probability values,
            or the one-hot labels. Must be float type. If the loss type is BCE, the shape of labels must be [N, C].

    Returns:
       Loss function to compute the loss between the input logits and labels.
    """
    name = name.lower()

    if name == "ce":
        loss = CrossEntropySmooth(smoothing=label_smoothing, aux_factor=aux_factor, reduction=reduction, weight=weight)
    elif name == "bce":
        loss = BinaryCrossEntropySmooth(
            smoothing=label_smoothing, aux_factor=aux_factor, reduction=reduction, weight=weight, pos_weight=None
        )
    elif name == "asl_single_label":
        loss = AsymmetricLossSingleLabel(smoothing=label_smoothing)
    elif name == "asl_multi_label":
        loss = AsymmetricLossMultilabel()
    elif name == "jsd":
        loss = JSDCrossEntropy(smoothing=label_smoothing, aux_factor=aux_factor, reduction=reduction, weight=weight)
    else:
        raise NotImplementedError

    return loss
