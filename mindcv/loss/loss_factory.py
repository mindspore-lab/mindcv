from mindspore import Tensor
from mindspore.nn import BCELoss, CrossEntropyLoss  # BCEWithLogitsLoss, SoftmaxCrossEntropyWithLogits
from .cross_entropy_smooth import CrossEntropySmooth
from typing import Optional


def create_loss(
        name: str = 'CE',
        weight: Optional[Tensor] = None,
        reduction: str = 'mean',
        label_smoothing: float = 0.,
        aux_factor: float = 0.):
    r"""
    create loss

    Args:
        name (str):  loss name, : 'BCE' (binary cross entropy), 'CE' for cross_entropy. Default: 'cross_entropy'
        weight (Tensor): The rescaling weight to each class. If the value is not None, the shape is (C,). 
            The data type only supports float32 or float16. Default: None.
            For bce loss, it is a manual rescaling weight given to the loss of each batch element. If given, has to be a Tensor of size nbatch.
        reduction (str):  Apply specific reduction method to the output: 'none', 'mean', or 'sum'.
            Default: 'mean'.
        label_smoothing (float): Label smoothing values, a regularization tool used to prevent the model
            from overfitting when calculating Loss. The value range is [0.0, 1.0]. Default value: 0.0.
        aux_factor: The factor for auxiluary loss, which is only applicable for cross entropy loss type for models with two logit outputs. 


    Inputs:(BCE)
        - **logits** (Tensor) - The input tensor with shape :math:`(N, *)` where :math:`*` means, any number of additional dimensions. The data type must be float16 or float32.
        - **labels** (Tensor) - The label tensor with shape :math:`(N, *)`, the same shape and data type as `logits`.


    Inputs:(CE)
        - **logits** (Tensor) - Tensor of shape :math:`(C,)` :math:`(N, C)` or :math:`(N, C, d_1, d_2, ..., d_K)` ,
          where `C = number of classes`. Data type must be float16 or float32.
        - **labels** (Tensor) - For class indices, tensor of shape :math:`()`, :math:`(N)` or :math:`(N, d_1, d_2, ..., d_K)` , data type must be int32.
          For probabilities, tensor of shape :math:`(C,)` :math:`(N, C)` or :math:`(N, C, d_1, d_2, ..., d_K)` , data type must be float16 or float32.



    Returns:
      created loss object, which will the computed loss value when invoked.  
    """

    if name.lower() == 'bce':
        # fixme: support label smoothing for BCE loss
        loss = BCELoss(weight=weight, reduction=reduction)
        if label_smoothing > 0:
            print(
                "Warning: Label smoothing is NOT effect because BCELoss does not support label smoothing for now.")  # fixme: use log package to output warning

    else:
        if aux_factor > 0:
            # fixme: 1) support reduction arg. 2) fixme: support class weight 
            loss = CrossEntropySmooth(smooth_factor=label_smoothing, factor=aux_factor)
        else:
            loss = CrossEntropyLoss(weight=weight, reduction=reduction, label_smoothing=label_smoothing)
            # loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    return loss
