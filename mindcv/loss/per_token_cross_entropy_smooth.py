""" cross entropy smooth """
from mindspore import nn
from mindspore.ops import functional as F


class PerTokenCrossEntropySmooth(nn.LossBase):
    """
    Per-token Cross entropy loss with label smoothing.
    Apply softmax activation function to input `logits`, and uses the given logits to compute cross entropy
    between the logits and the label.

    Args:
        smoothing: Label smoothing factor, a regularization tool used to prevent the model
            from overfitting when calculating Loss. The value range is [0.0, 1.0]. Default: 0.0.
        reduction: Apply specific reduction method to the output: 'mean' or 'sum'. Default: 'mean'.

    Inputs:
        logits (Tensor): Input logits. Shape [N, S, C], where N is # samples, S is # sequence length, C is # classes.
        labels (Tensor): Ground truth label. Shape: [N, S] or [N, S, C].
            (1) Shape (N, S), sparse labels representing the class indices. Must be int type.
            (2) Shape [N, S, C], dense labels representing the ground truth class probability values,
            or the one-hot labels. Must be float type.
    """

    def __init__(self, smoothing=0.0, reduction="mean"):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction
        self.ignore_index = -1

    def construct(self, logits, labels):
        logits = F.reshape(logits, (-1, logits.shape[-1]))
        if len(labels.shape) == 2:
            labels = F.reshape(labels, (-1,))
        else:
            labels = F.reshape(labels, (-1, labels.shape[-1]))

        loss = F.cross_entropy(
            logits, labels, ignore_index=self.ignore_index, reduction=self.reduction, label_smoothing=self.smoothing
        )
        return loss
