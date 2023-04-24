"""Ema define"""
import mindspore as ms
from mindspore import Parameter, Tensor, nn, ops
from mindspore.common import RowTensor
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P

from .gradient_accumulation import GradientAccumulation

__all__ = [
    "TrainStep",
]

_ema_op = C.MultitypeFuncGraph("grad_ema_op")
_grad_scale = C.MultitypeFuncGraph("grad_scale")
reciprocal = P.Reciprocal()
_grad_overflow = C.MultitypeFuncGraph("_grad_overflow")


@_ema_op.register("Tensor", "Tensor", "Tensor")
def _ema_weights(factor, ema_weight, weight):
    return F.assign(ema_weight, ema_weight * factor + weight * (1 - factor))


@_grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * F.cast(reciprocal(scale), F.dtype(grad))


@_grad_scale.register("Tensor", "RowTensor")
def tensor_grad_scale_row_tensor(scale, grad):
    return RowTensor(
        grad.indices,
        grad.values * F.cast(reciprocal(scale), F.dtype(grad.values)),
        grad.dense_shape,
    )


class TrainStep(nn.TrainOneStepWithLossScaleCell):
    """TrainStep with ema and clip grad."""

    def __init__(
        self,
        network,
        optimizer,
        scale_sense=1.0,
        ema=False,
        ema_decay=0.9999,
        updates=0,
        clip_grad=False,
        clip_value=15.0,
        gradient_accumulation_steps=1,
    ):
        super(TrainStep, self).__init__(network, optimizer, scale_sense)
        self.ema = ema
        self.ema_decay = ema_decay
        self.updates = Parameter(Tensor(updates, ms.float32))
        self.clip_grad = clip_grad
        self.clip_value = clip_value
        if self.ema:
            self.weights_all = ms.ParameterTuple(list(network.get_parameters()))
            self.ema_weight = self.weights_all.clone("ema", init="same")

        self.need_accumulate_grad = gradient_accumulation_steps > 1
        if self.need_accumulate_grad:
            self.gradient_accumulation = GradientAccumulation(gradient_accumulation_steps, optimizer, self.grad_reducer)

    def ema_update(self):
        """Update EMA parameters."""
        self.updates += 1
        d = self.ema_decay * (1 - F.exp(-self.updates / 2000))
        # update trainable parameters
        success = self.hyper_map(F.partial(_ema_op, d), self.ema_weight, self.weights_all)
        self.updates = F.depend(self.updates, success)
        return self.updates

    def construct(self, *inputs):
        """construct"""
        weights = self.weights
        loss = self.network(*inputs)
        scaling_sens = self.scale_sense

        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)

        scaling_sens_filled = C.ones_like(loss) * F.cast(scaling_sens, F.dtype(loss))
        grads = self.grad(self.network, weights)(*inputs, scaling_sens_filled)
        grads = self.hyper_map(F.partial(_grad_scale, scaling_sens), grads)
        if self.clip_grad:
            grads = ops.clip_by_global_norm(grads, clip_norm=self.clip_value)

        if self.need_accumulate_grad:
            # get the overflow buffer
            cond = self.get_overflow_status(status, grads)
            overflow = self.process_loss_scale(cond)
            loss = self.gradient_accumulation(loss, grads, overflow)
        else:
            # apply grad reducer on grads
            grads = self.grad_reducer(grads)
            loss = F.depend(loss, self.optimizer(grads))

        if self.ema:
            self.ema_update()
        return loss
