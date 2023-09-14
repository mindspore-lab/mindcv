import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor


class SoftmaxCrossEntropyLoss(nn.Cell):
    """
    softmax cross entropy loss
    """

    def __init__(self, num_cls=21, ignore_label=255):
        super(SoftmaxCrossEntropyLoss, self).__init__()
        self.cast = ops.Cast()
        self.one_hot = ops.OneHot(axis=-1)
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.ce = nn.SoftmaxCrossEntropyWithLogits()
        self.num_cls = num_cls
        self.ignore_label = ignore_label
        self.reduce_sum = ops.ReduceSum(False)

    def construct(self, logits, labels):
        labels_int = self.cast(labels, mstype.int32)
        labels_int = ops.reshape(labels_int, (-1,))
        logits_ = ops.transpose(logits, (0, 2, 3, 1))  # NCHW->NHWC
        logits_ = ops.reshape(logits_, (-1, self.num_cls))
        weights = ops.not_equal(labels_int, self.ignore_label)
        weights = self.cast(weights, mstype.float32)
        one_hot_labels = self.one_hot(labels_int, self.num_cls, self.on_value, self.off_value)
        loss = self.ce(logits_, one_hot_labels)
        loss = ops.mul(weights, loss)
        loss = ops.div(self.reduce_sum(loss), self.reduce_sum(weights))
        return loss
