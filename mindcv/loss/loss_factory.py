''' loss factory '''
from typing import Optional
from mindspore import Tensor
from mindspore.nn import BCELoss, CrossEntropyLoss
from .cross_entropy_smooth import CrossEntropySmooth

__all__ = ["create_loss"]


def create_loss(
        name: str = 'CE',
        weight: Optional[Tensor] = None,
        reduction: str = 'mean',
        label_smoothing: float = 0.,
        aux_factor: float = 0.):
    r"""Creates loss by name.

    Args:
        name:  loss name, : 'BCE' (binary cross entropy), 'CE' for cross_entropy. Default: 'CE'
        weight: The rescaling weight to each class. If the value is not None, the shape is (C,).
                The data type only supports float32 or float16. Default: None.
                For BCE Loss, a rescaling weight applied to the loss of each batch element. And it must have the same shape and data type as `inputs`. Default: None
        reduction: Apply specific reduction method to the output: 'none', 'mean', or 'sum'.
            Default: 'mean'.
        label_smoothing: Label smoothing values, a regularization tool used to prevent the model
            from overfitting when calculating Loss. The value range is [0.0, 1.0]. Default: 0.0.
        aux_factor: The factor for auxiliary loss. It should be set larger than 0 networks with two logit outputs (i.e., deep supervision), such as inception_v3.  Default: 0.0.

    Note:

    Inputs:
        if name is 'CE':
        - **logits** (Tensor) - Tensor of shape :math:`(C,)` :math:`(N, C)` or :math:`(N, C, d_1, d_2, ..., d_K)` ,
          where `C = number of classes`. Data type must be float16 or float32.
        - **labels** (Tensor) - For class indices, tensor of shape :math:`()`, :math:`(N)` or :math:`(N, d_1, d_2, ..., d_K)` , data type must be int32.
          For probabilities, tensor of shape :math:`(C,)` :math:`(N, C)` or :math:`(N, C, d_1, d_2, ..., d_K)` , data type must be float16 or float32.

        if name is 'BCE':
        - **logits** (Tensor) - The input tensor with shape :math:`(N, *)` where :math:`*` means, any number of additional dimensions. The data type must be float16 or float32.
        - **labels** (Tensor) - The label tensor with shape :math:`(N, *)`, the same shape and data type as `logits`.

    Returns:
       Loss object that will be invoked to computed loss value.
    """

    if name.lower() == 'bce':
        # fixme: support label smoothing for BCE loss
        loss = BCELoss(weight=weight, reduction=reduction)
        if label_smoothing > 0:
            print("Warning: Label smoothing is NOT effect because BCELoss does not support label smoothing for now.")  # fixme: use log package to output warning
        if aux_factor > 0:
            print("Warning: Aux factor is NOT effect because BCELoss does not support aux factor.")  # fixme: use log package to output warning

    else:
        if aux_factor > 0:
            # fixme: 1) support reduction arg. 2) fixme: support class weight
            loss = CrossEntropySmooth(smooth_factor=label_smoothing, factor=aux_factor)
            if weight is not None:
                print("Warning: weight is NOT effect because CrossEntropySmooth does not class weight.")
        else:
            loss = CrossEntropyLoss(weight=weight, reduction=reduction, label_smoothing=label_smoothing)
            # loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    return loss
