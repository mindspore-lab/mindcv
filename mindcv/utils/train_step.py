"""Customized TrainOneStepCell.

Supported algorithms are list as follows:
    * Exponential Moving Average (EMA)
    * Gradient Clipping
    * Gradient Accumulation
"""

import mindspore as ms
from mindspore import Parameter, RowTensor, Tensor, boost, nn, ops
from mindspore.boost.grad_accumulation import gradient_accumulation_op, gradient_clear_op
from mindspore.ops import functional as F

__all__ = [
    "GradientAccumulation",
    "TrainStep",
]

_ema_op = ops.MultitypeFuncGraph("ema_op")
_grad_scale = ops.MultitypeFuncGraph("grad_scale")
reciprocal = ops.Reciprocal()


@_ema_op.register("Tensor", "Tensor", "Tensor")
def ema_ops(factor, ema_weight, weight):
    return F.assign(ema_weight, ema_weight * factor + weight * (1 - factor))


@_grad_scale.register("Tensor", "Tensor")
def grad_scale_tensor(scale, grad):
    return grad * F.cast(reciprocal(scale), F.dtype(grad))


@_grad_scale.register("Tensor", "RowTensor")
def grad_scale_row_tensor(scale, grad):
    return RowTensor(
        grad.indices,
        grad.values * F.cast(reciprocal(scale), F.dtype(grad.values)),
        grad.dense_shape,
    )


class GradientAccumulation(boost.GradientAccumulation):
    """
    After accumulating the gradients of multiple steps, call to optimize its update.

    Note:
        The implementation is based on mindspore.boost.GradientAccumulation with the following modifications:

        1. The learning rate will be updated at each iteration step.
            However, in the original implementation, the learning rate will only be updated for each M iteration steps.
        2. For distributed training, at gradient accumulation stage, each device will maintain
            its own accumulated gradient; At the parameter update stage, the gradient will be synchronized
            across the devices first and then perform the gradient update.

    Args:
       max_accumulation_step (int): Steps to accumulate gradients.
       optimizer (nn.Cell): Optimizer used.
       grad_reducer (nn.Cell): Gradient reducer, which synchronize gradients across the devices.
    """

    def __init__(self, max_accumulation_step, optimizer, grad_reducer):
        super().__init__(max_accumulation_step, optimizer)
        self.grad_reducer = grad_reducer

    def construct(self, loss, grads):
        loss = F.depend(
            loss,
            self.hyper_map(
                F.partial(gradient_accumulation_op, self._max_accumulation_step), self._grad_accumulation, grads
            ),
        )
        self._accumulation_step += 1

        if self._accumulation_step >= self._max_accumulation_step:
            # accumulate the gradient at each device, don't sync them until updating the weight
            reduced_grad_accumulation = self.grad_reducer(self._grad_accumulation)
            loss = F.depend(loss, self.optimizer(reduced_grad_accumulation))
            loss = F.depend(loss, self.hyper_map(F.partial(gradient_clear_op), self._grad_accumulation))
            self._accumulation_step = 0
        else:
            # update the learning rate, do not update the parameter
            loss = F.depend(loss, self.optimizer.get_lr())

        return loss


class TrainStep(nn.TrainOneStepWithLossScaleCell):
    """Training step with loss scale.

    The customized trainOneStepCell also supported following algorithms:
        * Exponential Moving Average (EMA)
        * Gradient Clipping
        * Gradient Accumulation
    """

    def __init__(
        self,
        network,
        optimizer,
        scale_sense=1.0,
        ema=False,
        ema_decay=0.9999,
        clip_grad=False,
        clip_value=15.0,
        gradient_accumulation_steps=1,
    ):
        super(TrainStep, self).__init__(network, optimizer, scale_sense)
        self.ema = ema
        self.ema_decay = ema_decay
        self.updates = Parameter(Tensor(0.0, ms.float32))
        self.clip_grad = clip_grad
        self.clip_value = clip_value
        if self.ema:
            self.weights_all = ms.ParameterTuple(list(network.get_parameters()))
            self.ema_weight = self.weights_all.clone("ema", init="same")

        self.accumulate_grad = gradient_accumulation_steps > 1
        if self.accumulate_grad:
            self.gradient_accumulation = GradientAccumulation(gradient_accumulation_steps, optimizer, self.grad_reducer)

    def ema_update(self):
        self.updates += 1
        # ema factor is corrected by (1 - exp(-t/T)), where `t` means time and `T` means temperature.
        ema_decay = self.ema_decay * (1 - F.exp(-self.updates / 2000))
        # update trainable parameters
        success = self.hyper_map(F.partial(_ema_op, ema_decay), self.ema_weight, self.weights_all)
        return success

    def construct(self, *inputs):
        weights = self.weights
        loss = self.network(*inputs)
        scaling_sens = self.scale_sense

        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)

        scaling_sens_filled = ops.ones_like(loss) * F.cast(scaling_sens, F.dtype(loss))
        grads = self.grad(self.network, weights)(*inputs, scaling_sens_filled)
        grads = self.hyper_map(F.partial(_grad_scale, scaling_sens), grads)

        # todo: When to clip grad? Do we need to clip grad after grad reduction? What if grad accumulation is needed?
        if self.clip_grad:
            grads = ops.clip_by_global_norm(grads, clip_norm=self.clip_value)

        if self.loss_scaling_manager:  # scale_sense = update_cell: Cell --> TrainOneStepWithLossScaleCell.construct
            if self.accumulate_grad:
                # todo: GradientAccumulation only call grad_reducer at the step where the accumulation is completed.
                #  So checking the overflow status is after gradient reduction, is this correct?
                # get the overflow buffer
                cond = self.get_overflow_status(status, grads)
                overflow = self.process_loss_scale(cond)
                # if there is no overflow, do optimize
                if not overflow:
                    loss = self.gradient_accumulation(loss, grads)
                    if self.ema:
                        loss = F.depend(loss, self.ema_update())
            else:
                # apply grad reducer on grads
                grads = self.grad_reducer(grads)
                # get the overflow buffer
                cond = self.get_overflow_status(status, grads)
                overflow = self.process_loss_scale(cond)
                # if there is no overflow, do optimize
                if not overflow:
                    loss = F.depend(loss, self.optimizer(grads))
                    if self.ema:
                        loss = F.depend(loss, self.ema_update())
        else:  # scale_sense = loss_scale: Tensor --> TrainOneStepCell.construct
            if self.accumulate_grad:
                loss = self.gradient_accumulation(loss, grads)
            else:
                grads = self.grad_reducer(grads)
                loss = F.depend(loss, self.optimizer(grads))

            if self.ema:
                loss = F.depend(loss, self.ema_update())

        return loss
