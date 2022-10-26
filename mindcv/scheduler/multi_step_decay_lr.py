"""MultiStep Decay Learning Rate Scheduler"""
import mindspore as ms
from mindspore.nn.learning_rate_schedule import LearningRateSchedule

class MultiStepDecayLR(LearningRateSchedule):
    """ Multiple step learning rate
    The learning rate will decay once the number of step reaches one of the milestones.
    """

    def __init__(self, lr, decay_rate, milestones, steps_per_epoch, num_epochs):
        super().__init__()
        num_steps = num_epochs * steps_per_epoch
        step_lrs = []
        cur_lr = lr
        k = 0
        for step in range(num_steps):
            if step == milestones[k] * steps_per_epoch :
                cur_lr = cur_lr * decay_rate
                k = min(k+1, len(milestones)-1)
            step_lrs.append(cur_lr)

        self.step_lrs = ms.Tensor(step_lrs, ms.float32)
        #print(self.step_lrs)

    def construct(self, global_step):
        if global_step < self.step_lrs.shape[0]:
            lr = self.step_lrs[global_step]
        else:
            lr = self.step_lrs[-1]
        return lr
