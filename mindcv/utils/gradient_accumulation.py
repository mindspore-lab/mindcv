from mindspore.common import Parameter, Tensor
from mindspore.common import dtype as mstype
from mindspore.nn.cell import Cell
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P

__all__ = ["GradientAccumulation", "gradient_accumulation_op", "gradient_clear_op"]


gradient_accumulation_op = C.MultitypeFuncGraph("gradient_accumulation_op")


@gradient_accumulation_op.register("Int64", "Tensor", "Tensor")
def cumulative_grad_process(accumulation_step, cumulative_grad, grad):
    """Apply gradient accumulation to cumulative grad."""
    return P.AssignAdd()(cumulative_grad, grad / accumulation_step)


gradient_clear_op = C.MultitypeFuncGraph("gradient_clear_op")


@gradient_clear_op.register("Tensor")
def clear_grad(cumulative_grad):
    """Clear grad."""
    zero_grad = P.ZerosLike()(cumulative_grad)
    return F.assign(cumulative_grad, zero_grad)


class GradientAccumulation(Cell):
    """
    After accumulating the gradients of multiple steps, call to optimize its update.

    Args:
       max_accumulation_step (int): Steps to accumulate gradients.
       optimizer (Cell): Optimizer used.
    """

    def __init__(self, max_accumulation_step, optimizer, grad_reducer):
        super(GradientAccumulation, self).__init__()
        self._max_accumulation_step = max_accumulation_step
        self.optimizer = optimizer
        self.weights = optimizer.parameters
        self.hyper_map = C.HyperMap()
        self.grad_reducer = grad_reducer
        self._grad_accumulation = self.weights.clone(prefix="grad_accumulation", init="zeros")
        self._accumulation_step = Parameter(Tensor(0, dtype=mstype.int32), name="accumulation_step")

    def construct(self, loss, grads, overflow):
        # do not accumulate the grad it it is overflow
        if overflow:
            return loss

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
