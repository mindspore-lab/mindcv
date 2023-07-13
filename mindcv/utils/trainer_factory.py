import logging
from typing import Optional, Union

import mindspore as ms
from mindspore import Tensor, context
from mindspore import dtype as mstype
from mindspore import nn
from mindspore.ops import functional as F
from mindspore.train import DynamicLossScaleManager, FixedLossScaleManager, Model

from .amp import auto_mixed_precision
from .train_step import TrainStep

__all__ = [
    "get_metrics",
    "require_customized_train_step",
    "create_trainer",
]

_logger = logging.getLogger(__name__)


def get_metrics(num_classes):
    if num_classes >= 5:
        metrics = {
            "Top_1_Accuracy": nn.Top1CategoricalAccuracy(),
            "Top_5_Accuracy": nn.Top5CategoricalAccuracy(),
        }
    else:
        metrics = {
            "Top_1_Accuracy": nn.Top1CategoricalAccuracy(),
        }
    return metrics


def require_customized_train_step(
    ema: bool = False,
    clip_grad: bool = False,
    gradient_accumulation_steps: int = 1,
    amp_cast_list: Optional[str] = None,
):
    if ema:
        return True
    if clip_grad:
        return True
    if gradient_accumulation_steps > 1:
        return True
    if amp_cast_list:
        return True
    return False


def add_loss_network(network, loss_fn, amp_level):
    """Add loss network."""

    class WithLossCell(nn.Cell):
        "Wrap loss for amp. Cast network output back to float32"

        def __init__(self, backbone, loss_fn):
            super(WithLossCell, self).__init__(auto_prefix=False)
            self._backbone = backbone
            self._loss_fn = loss_fn

        def construct(self, data, label):
            out = self._backbone(data)
            label = F.mixed_precision_cast(mstype.float32, label)
            return self._loss_fn(F.mixed_precision_cast(mstype.float32, out), label)

    if amp_level == "O2" or amp_level == "O3":
        network = WithLossCell(network, loss_fn)
    else:
        network = nn.WithLossCell(network, loss_fn)
    return network


def create_trainer(
    network: nn.Cell,
    loss: nn.Cell,
    optimizer: nn.Cell,
    metrics: Union[dict, set],
    amp_level: str,
    amp_cast_list: str,
    loss_scale_type: str,
    loss_scale: float = 1.0,
    drop_overflow_update: bool = False,
    ema: bool = False,
    ema_decay: float = 0.9999,
    clip_grad: bool = False,
    clip_value: float = 15.0,
    gradient_accumulation_steps: int = 1,
):
    """Create Trainer.

    Args:
        network: The backbone network to train, evaluate or predict.
        loss: The function of calculating loss.
        optimizer: The optimizer for training.
        metrics: The metrics for model evaluation.
        amp_level: The level of auto mixing precision training.
        amp_cast_list: At the cell level, custom casting the cell to FP16.
        loss_scale_type: The type of loss scale.
        loss_scale: The value of loss scale.
        drop_overflow_update: Whether to execute optimizer if there is an overflow.
        ema: Whether to use exponential moving average of model weights.
        ema_decay: Decay factor for model weights moving average.
        clip_grad: whether to gradient clip.
        clip_value: The value at which to clip gradients.
        gradient_accumulation_steps: Accumulate the gradients of n batches before update.

    Returns:
        mindspore.Model

    """
    if loss_scale < 1.0:
        raise ValueError("Loss scale cannot be less than 1.0!")

    if drop_overflow_update is False and loss_scale_type.lower() == "dynamic":
        raise ValueError("DynamicLossScale ALWAYS drop overflow!")

    if gradient_accumulation_steps < 1:
        raise ValueError("`gradient_accumulation_steps` must be >= 1!")

    if not require_customized_train_step(ema, clip_grad, gradient_accumulation_steps, amp_cast_list):
        mindspore_kwargs = dict(
            network=network,
            loss_fn=loss,
            optimizer=optimizer,
            metrics=metrics,
            amp_level=amp_level,
        )
        if loss_scale_type.lower() == "fixed":
            mindspore_kwargs["loss_scale_manager"] = FixedLossScaleManager(
                loss_scale=loss_scale, drop_overflow_update=drop_overflow_update
            )
        elif loss_scale_type.lower() == "dynamic":
            mindspore_kwargs["loss_scale_manager"] = DynamicLossScaleManager(
                init_loss_scale=loss_scale, scale_factor=2, scale_window=2000
            )
        elif loss_scale_type.lower() == "auto":
            # We don't explicitly construct LossScaleManager
            _logger.warning(
                "You are using AUTO loss scale, which means the LossScaleManager isn't explicitly pass in "
                "when creating a mindspore.Model instance. "
                "NOTE: mindspore.Model may use LossScaleManager silently. See mindspore.train.amp for details."
            )
        else:
            raise ValueError(f"Loss scale type only support ['fixed', 'dynamic', 'auto'], but got{loss_scale_type}.")
        model = Model(**mindspore_kwargs)
    else:  # require customized train step
        eval_network = nn.WithEvalCell(network, loss, amp_level in ["O2", "O3", "auto"])
        auto_mixed_precision(network, amp_level, amp_cast_list)
        net_with_loss = add_loss_network(network, loss, amp_level)
        train_step_kwargs = dict(
            network=net_with_loss,
            optimizer=optimizer,
            ema=ema,
            ema_decay=ema_decay,
            clip_grad=clip_grad,
            clip_value=clip_value,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
        if loss_scale_type.lower() == "fixed":
            loss_scale_manager = FixedLossScaleManager(loss_scale=loss_scale, drop_overflow_update=drop_overflow_update)
        elif loss_scale_type.lower() == "dynamic":
            loss_scale_manager = DynamicLossScaleManager(init_loss_scale=loss_scale, scale_factor=2, scale_window=2000)
        else:
            raise ValueError(f"Loss scale type only support ['fixed', 'dynamic'], but got{loss_scale_type}.")
        update_cell = loss_scale_manager.get_update_cell()
        # 1. loss_scale_type="fixed", drop_overflow_update=False
        # --> update_cell=None, TrainStep=TrainOneStepCell(scale_sense=loss_scale)
        # 2. loss_scale_type: fixed, drop_overflow_update: True
        # --> update_cell=FixedLossScaleUpdateCell, TrainStep=TrainOneStepWithLossScaleCell(scale_sense=update_cell)
        # 3. loss_scale_type: dynamic, drop_overflow_update: True
        # --> update_cell=DynamicLossScaleUpdateCell, TrainStep=TrainOneStepWithLossScaleCell(scale_sense=update_cell)
        if update_cell is None:
            train_step_kwargs["scale_sense"] = Tensor(loss_scale, dtype=ms.float32)
        else:
            if not context.get_context("enable_ge") and context.get_context("device_target") == "CPU":
                raise ValueError(
                    "Only `loss_scale_type` is `fixed` and `drop_overflow_update` is `False`"
                    "are supported on device `CPU`."
                )
            train_step_kwargs["scale_sense"] = update_cell
        train_step_cell = TrainStep(**train_step_kwargs).set_train()
        model = Model(train_step_cell, eval_network=eval_network, metrics=metrics, eval_indexes=[0, 1, 2])
        # todo: do we need to set model._loss_scale_manager
    return model
