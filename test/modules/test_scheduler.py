import sys
sys.path.append('.')

import pytest

from mindcv.scheduler import create_scheduler
import numpy as np
from mindspore import Tensor
import mindspore as ms
import mindspore.nn as nn
from mindspore.common.initializer import Normal

@pytest.mark.parametrize('sched', ['polynomial_decay', 'warmup_cosine_decay', 'exponential_decay', 'step_decay', 'constant'])
def test_scheduler(sched):
    
    learning_rate = lr = 0.1
    min_lr = end_learning_rate = 0.01
    decay_rate = power = 0.5
    steps_per_epoch = 10
    decay_epochs = 1

    #polynomial_decay_lr = nn.PolynomialDecayLR(learning_rate, end_learning_rate, decay_steps, power)
    if 'warmup' in sched:
        warmup_epochs = 1
    else:
        warmup_epochs = 0

    scheduler = create_scheduler(steps_per_epoch, sched, lr=lr, min_lr=min_lr, warmup_epochs=warmup_epochs,decay_epochs=decay_epochs, decay_rate=decay_rate)
    
    # TODO: check warmup 

    # check decay trend 
    cur_epoch = 2
    global_step = Tensor(cur_epoch * steps_per_epoch, ms.int32)

    
    if sched not in ['constant']:
        cur_lr = scheduler(global_step)
        print(cur_lr)
        assert cur_lr < lr, 'LR decay error'
    else:
        cur_lr = scheduler
        assert cur_lr==lr, 'LR constant error'


    # TODO: check value correctness
