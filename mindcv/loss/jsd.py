from mindspore import nn, ops

from .cross_entropy_smooth import CrossEntropySmooth


class JSDCrossEntropy(nn.LossBase):
    """
    JSD loss is implemented according to "AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty"
    https://arxiv.org/abs/1912.02781

    Please note that JSD loss should be used when "aug_splits = 3".
    """

    def __init__(self, num_splits=3, alpha=12, smoothing=0.1, weight=None, reduction="mean", aux_factor=0.0):
        super().__init__()
        self.num_splits = num_splits
        self.alpha = alpha
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction
        self.kldiv = ops.KLDivLoss(reduction="batchmean")
        self.map = ops.Map()

        self.softmax = ops.Softmax(axis=1)
        self.aux_factor = aux_factor

    def construct(self, logits, labels):
        if self.training:
            split_size = logits.shape[0] // self.num_splits
            log_split = ops.split(logits, 0, self.num_splits)

            loss = ops.cross_entropy(
                log_split[0],
                labels[:split_size],
                weight=self.weight,
                reduction=self.reduction,
                label_smoothing=self.smoothing,
            )

            probs = self.map(self.softmax, log_split)
            stack_probs = ops.stack(probs)
            clip_probs = ops.clip_by_value(stack_probs.mean(axis=0), 1e-7, 1)
            log_probs = ops.log(clip_probs)

            for p_split in probs:
                loss += self.alpha * self.kldiv(log_probs, p_split) / self.num_splits

            return loss
        else:
            return CrossEntropySmooth(
                smoothing=self.smoothing, aux_factor=self.aux_factor, reduction=self.reduction, weight=self.weight
            )(logits, labels)
