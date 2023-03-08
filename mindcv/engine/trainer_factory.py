import logging
from typing import Union

import mindspore as ms
from mindspore import nn
from mindspore.train import DynamicLossScaleManager, FixedLossScaleManager, Model

from .train_step import TrainStep

__all__ = [
    "get_metrics",
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


def _require_customized_train_step(ema: bool = False, clip_grad: bool = False):
    if ema:
        return True
    if clip_grad:
        return True
    return False


def create_trainer(
    network: nn.Cell,
    loss: nn.Cell,
    optimizer: nn.Cell,
    metrics: Union[dict, set],
    amp_level: str,
    loss_scale_type: str,
    loss_scale: float = 1.0,
    drop_overflow_update: bool = False,
    ema: bool = False,
    ema_decay: float = 0.9999,
    clip_grad: bool = False,
    clip_value: float = 15.0,
):
    """Create Trainer.

    Args:
        network: The backbone network to train, evaluate or predict.
        loss: The function of calculating loss.
        optimizer: The optimizer for training.
        metrics: The metrics for model evaluation.
        amp_level: The level of auto mixing precision training.
        loss_scale_type: The type of loss scale.
        loss_scale: The value of loss scale.
        drop_overflow_update: Whether to execute optimizer if there is an overflow.
        ema: Whether to use exponential moving average of model weights.
        ema_decay: Decay factor for model weights moving average.
        clip_grad: whether to gradient clip.
        clip_value: The value at which to clip gradients.

    Returns:
        mindspore.Model

    """
    if loss_scale < 1.0:
        raise ValueError("Loss scale cannot be less than 1.0!")

    if drop_overflow_update is False and loss_scale_type.lower() == "dynamic":
        raise ValueError("DynamicLossScale ALWAYS drop overflow!")

    if not _require_customized_train_step(ema, clip_grad):
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
        net_with_loss = nn.WithLossCell(network, loss)
        ms.amp.auto_mixed_precision(net_with_loss, amp_level=amp_level)
        train_step_kwargs = dict(
            network=net_with_loss,
            optimizer=optimizer,
            ema=ema,
            ema_decay=ema_decay,
            clip_grad=clip_grad,
            clip_value=clip_value,
        )
        if loss_scale_type.lower() == "fixed":
            # todo: drop_overflow_update. If drop_overflow_update is False, scale_sense should be a number
            #  instead of cell, and TrainStep should be TrainOneStepCell. If drop_overflow_update is True,
            #  scale_sense should be FixedLossScaleUpdateCell, and TrainStep should be TrainOneStepWithLossScaleCell.
            train_step_kwargs["scale_sense"] = nn.FixedLossScaleUpdateCell(loss_scale_value=loss_scale)
        elif loss_scale_type.lower() == "dynamic":
            train_step_kwargs["scale_sense"] = nn.DynamicLossScaleUpdateCell(
                loss_scale_value=loss_scale, scale_factor=2, scale_window=2000
            )
        else:
            raise ValueError(f"Loss scale type only support ['fixed', 'dynamic'], but got{loss_scale_type}.")
        # todo: remove this check when TrainStep support dynamic loss scale and dropping overflow
        if drop_overflow_update or loss_scale_type.lower() != "fixed":
            raise ValueError("TrainStep only support fixed loss scale without dropping overflow!")
        train_step_cell = TrainStep(**train_step_kwargs)
        eval_network = nn.WithEvalCell(network, loss, amp_level in ["O2", "O3", "auto"])
        model = Model(train_step_cell, eval_network=eval_network, metrics=metrics, eval_indexes=[0, 1, 2])
        # todo: do we need to set model._loss_scale_manager
    return model
