''' cross entorpy smooth '''
from mindspore import nn
from mindspore import ops


class CrossEntropySmooth(nn.LossBase):
    '''cross entropy loss with label smoothing. '''
    def __init__(self, smooth_factor=0., factor=0.):
        super().__init__()
        self.smoothing = smooth_factor
        self.confidence = 1. - smooth_factor
        self.factor = factor
        self.log_softmax = ops.LogSoftmax()
        self.gather = ops.Gather()
        self.expand = ops.ExpandDims()

    def construct(self, logits, labels):
        loss_aux = 0
        if self.factor > 0:
            logits, aux = logits
            auxprobs = self.log_softmax(aux)
            nll_loss_aux = ops.gather_d((-1 * auxprobs), 1, self.expand(labels, -1))
            nll_loss_aux = nll_loss_aux.squeeze(1)
            smooth_loss = -auxprobs.mean(axis=-1)
            loss_aux = (self.confidence * nll_loss_aux + self.smoothing * smooth_loss).mean()
        logprobs = self.log_softmax(logits)
        nll_loss_logits = ops.gather_d((-1 * logprobs), 1, self.expand(labels, -1))
        nll_loss_logits = nll_loss_logits.squeeze(1)
        smooth_loss = -logprobs.mean(axis=-1)
        loss_logits = (self.confidence * nll_loss_logits + self.smoothing * smooth_loss).mean()
        loss = loss_logits + self.factor * loss_aux
        return loss
