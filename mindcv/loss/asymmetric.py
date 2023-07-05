import numpy as np

import mindspore.nn as nn
from mindspore import Tensor, ops


class AsymmetricLossMultilabel(nn.LossBase):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super(AsymmetricLossMultilabel, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def construct(self, logits, labels):
        """
        logits: output from models
        labels: multi-label binarized vector
        """
        x_sigmoid = ops.Sigmoid()(logits)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        if self.clip > 0:
            xs_neg = ops.clip_by_value(xs_neg + self.clip, clip_value_max=Tensor(1.0))

        los_pos = labels * ops.log(ops.clip_by_value(xs_pos, clip_value_min=Tensor(self.eps)))
        los_neg = (1 - labels) * ops.log(ops.clip_by_value(xs_neg, clip_value_min=Tensor(self.eps)))

        loss = los_pos + los_neg

        if self.gamma_pos > 0 and self.gamma_neg > 0:
            pt0 = xs_pos * labels
            pt1 = xs_neg * (1 - labels)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * labels + self.gamma_neg * (1 - labels)
            one_sided_w = ops.pow(1 - pt, one_sided_gamma)

            loss *= one_sided_w

        return -loss.sum()


class AsymmetricLossSingleLabel(nn.LossBase):
    def __init__(self, gamma_pos=1, gamma_neg=4, eps=0.1, reduction="mean", smoothing=0.1):
        super(AsymmetricLossSingleLabel, self).__init__()

        self.eps = eps
        self.logsoftmax = nn.LogSoftmax(axis=-1)
        self.targets_classes = []
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.reduction = reduction
        self.smoothing = smoothing

    def construct(self, logits, labels):
        num_classes = logits.shape[-1]
        log_preds = self.logsoftmax(logits)
        labels_e = ops.ExpandDims()(labels, 1)
        labels_e_shape = labels_e.shape
        targets = ops.tensor_scatter_elements(
            ops.ZerosLike()(logits), labels_e, Tensor(np.ones(labels_e_shape, dtype=np.float32)), 1
        )

        anti_targets = 1 - targets
        xs_pos = ops.exp((log_preds))
        xs_neg = 1 - xs_pos
        xs_pos = xs_pos * targets
        xs_neg = xs_neg * anti_targets

        asymmetric_w = ops.pow(1 - xs_pos - xs_neg, self.gamma_pos * targets + self.gamma_neg * anti_targets)

        log_preds = log_preds * asymmetric_w

        targets = targets * (1 - self.smoothing) + self.smoothing / num_classes

        loss = -targets * log_preds
        loss = ops.ReduceSum()(loss, -1)

        if self.reduction == "mean":
            loss = loss.mean()

        return loss
