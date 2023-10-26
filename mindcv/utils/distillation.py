""" distillation related functions """
from types import MethodType

import mindspore as ms
from mindspore import nn
from mindspore.ops import functional as F


class DistillLossCell(nn.WithLossCell):
    """
    Wraps the network with hard distillation loss function.

    Get the loss of student network and an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.

    Args:
        backbone (Cell): The student network to train and calculate base loss.
        loss_fn (Cell): The loss function used to compute loss of student network.
        distillation_type (str): The type of distillation.
        teacher_model (Cell): The teacher network to calculate distillation loss.
        alpha (float): The coefficient to balance the distillation loss and base loss. Default: 0.5.
        tau (float): Distillation temperature. The higher the temperature, the lower the
            dispersion of the loss calculated by Kullback-Leibler divergence loss. Default: 1.0.
    """

    def __init__(self, backbone, loss_fn, distillation_type, teacher_model, alpha=0.5, tau=1.0):
        super().__init__(backbone, loss_fn)
        if distillation_type == "hard":
            self.hard_type = True
        elif distillation_type == "soft":
            self.hard_type = False
        else:
            raise ValueError(f"Distillation type only support ['hard', 'soft'], but got {distillation_type}.")
        self.teacher_model = teacher_model
        self.alpha = alpha
        self.tau = tau

    def construct(self, data, label):
        out = self._backbone(data)

        out, out_kd = out
        base_loss = self._loss_fn(out, label)

        teacher_out = F.stop_gradient(self.teacher_model(data))

        if self.hard_type:
            distillation_loss = F.cross_entropy(out_kd, teacher_out.argmax(axis=1))
        else:
            T = self.tau
            out_kd = F.cast(out_kd, ms.float32)
            distillation_loss = (
                F.kl_div(
                    F.log_softmax(out_kd / T, axis=1),
                    F.log_softmax(teacher_out / T, axis=1),
                    reduction="sum",
                )
                * (T * T)
                / F.size(out_kd)
            )

        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha

        return loss


def bn_infer_only(self, x):
    return self.bn_infer(x, self.gamma, self.beta, self.moving_mean, self.moving_variance)[0]


def dropout_infer_only(self, x):
    return x


def set_validation(network):
    """
    Since MindSpore cannot automatically set some cells to validation mode
    during training in the teacher network, we need to manually set these
    cells to validation mode in this function.
    """

    for _, cell in network.cells_and_names():
        if isinstance(cell, (nn.BatchNorm2d, nn.BatchNorm1d, nn.BatchNorm3d)):
            cell.construct = MethodType(bn_infer_only, cell)
        elif isinstance(cell, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            cell.construct = MethodType(dropout_infer_only, cell)
        else:
            cell.set_train(False)
