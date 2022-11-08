"""Cosine Decay with Warmup Learning Rate Scheduler"""
from mindspore import nn, ops
from mindspore.nn.learning_rate_schedule import LearningRateSchedule



class WarmupCosineDecayLR(LearningRateSchedule):
    """ CosineDecayLR with warmup
    The learning rate will increase from 0 to max_lr in `warmup_epochs` epochs, then decay to min_lr in `decay_epoches` epochs
    """

    def __init__(self,
                 min_lr,
                 max_lr,
                 warmup_epochs,
                 decay_epochs,
                 steps_per_epoch
                 ):
        super().__init__()
        self.warmup_steps = warmup_epochs * steps_per_epoch
        self.decay_steps = decay_epochs * steps_per_epoch
        if self.warmup_steps > 0:
            self.warmup_lr = nn.WarmUpLR(max_lr, self.warmup_steps)
        self.cosine_decay_lr = nn.CosineDecayLR(min_lr, max_lr, self.decay_steps)

    def construct(self, global_step):
        if self.warmup_steps > 0:
            if global_step > self.warmup_steps:
                lr = self.cosine_decay_lr(global_step - self.warmup_steps)
            else:
                lr = self.warmup_lr(global_step)
        else:
            lr = self.cosine_decay_lr(global_step)
        return lr
