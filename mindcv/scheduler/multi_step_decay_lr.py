"""MultiStep Decay Learning Rate Scheduler"""
from mindspore import nn
from mindspore.nn.learning_rate_schedule import LearningRateSchedule


class MultiStepDecayLR(LearningRateSchedule):
    """ Multiple step learning rate
    The learning rate will decay once the number of step reaches one of the milestones.
    """

    def __init__(self, learning_rate, decay_rate, decay_step_indices):
        super().__init__()
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.decay_step_indices = decay_step_indices
        self.step_lrs = [learning_rate * decay_rate ** i for i, m in enumerate(decay_step_indices)]
        self.piecewise_constant_lr = nn.piecewise_constant_lr(self.decay_step_indices, self.step_lrs)

    def construct(self, global_step):

        if global_step < self.decay_step_indices[-1]:
            lr = self.piecewise_constant_lr[global_step]
        else:
            lr = self.step_lrs[-1] * self.decay_rate
        return lr
