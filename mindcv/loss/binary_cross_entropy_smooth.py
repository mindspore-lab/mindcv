''' cross entorpy smooth '''
#import warnings
from mindspore import nn
from mindspore.ops import functional as F
from mindspore.ops import operations as P

class BinaryCrossEntropySmooth(nn.LossBase):
    '''
    Binary cross entropy loss with label smoothing.
    Apply sigmoid activation function to input `logits`, and uses the given logits to compute binary cross entropy
    between the logits and the label.

    Args:
        smoothing: Label smoothing factor, a regularization tool used to prevent the model
            from overfitting when calculating Loss. The value range is [0.0, 1.0]. Default: 0.0.
        aux_factor: Auxiliary loss factor. Set aux_fuactor > 0.0 if the model has auxilary logit outputs (i.e., deep supervision), like inception_v3.  Default: 0.0.
        reduction: Apply specific reduction method to the output: 'mean' or 'sum'. Default: 'mean'.
        weight (Tensor): Class weight. A rescaling weight applied to the loss of each batch element. Shape [C]. It can be
          broadcast to a tensor with shape of `logits`. Data type must be float16 or float32.
        pos_weight (Tensor): Positive weight for each class. A weight of positive examples. Shape [C]. Must be a vector with length equal to the
          number of classes. It can be broadcast to a tensor with shape of `logits`. Data type must be float16 or float32.

    Inputs:
        logits (Tensor or Tuple of Tensor): (1) Input logits. Shape [N, C], where N is # samples, C is # classes.
                ,or (2) Tuple of two input logits (main_logits and aux_logits) for auxilary loss.
        labels (Tensor): Ground truth label, shape [N, C], has the same shape as `logits`. can be a class probability matrix or one-hot labels.
          Data type must be float16 or float32.
    '''
    def __init__(self, smoothing=0., aux_factor=0., reduction='mean', weight=None, pos_weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.aux_factor = aux_factor
        self.reduction = reduction
        self.weight = weight
        self.pos_weight = pos_weight
        self.ones = P.OnesLike()

    def construct(self, logits, labels):
        loss_aux = 0
        aux_logits = None

        if isinstance(logits, tuple):
            main_logits = logits[0]
        else:
            main_logits = logits

        ones_input = self.ones(main_logits)
        if self.weight is not None:
            weight = self.weight
        else:
            weight = ones_input
        if self.pos_weight is not None:
            pos_weight = self.pos_weight
        else:
            pos_weight = ones_input

        if self.smoothing > 0.0:
            class_dim = 0 if main_logits.ndim == 1 else -1
            n_classes = main_logits.shape[class_dim]
            labels = labels * (1 - self.smoothing) + self.smoothing / n_classes

        if self.aux_factor > 0 and aux_logits is not None:
            for aux_logits in logits[1:]:
                loss_aux += F.binary_cross_entropy_with_logits(aux_logits, labels, weight=weight, pos_weight=pos_weight, reduction=self.reduction)
        # else:
        #    warnings.warn("There are logit tuple input, but the auxilary loss factor is 0.")

        loss_logits = F.binary_cross_entropy_with_logits(main_logits, labels, weight=weight, pos_weight=pos_weight, reduction=self.reduction)

        loss = loss_logits + self.aux_factor * loss_aux

        return loss
