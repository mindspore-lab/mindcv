import sys
sys.path.append('.')

import pytest

from mindcv.scheduler import create_scheduler
import numpy as np
from mindspore import Tensor
import mindspore as ms
import mindspore.nn as nn
from mindspore.common.initializer import Normal


@pytest.mark.parametrize('milestones', [[1, 3, 5, 7], [2, 4, 6, 8]])
@pytest.mark.parametrize('warmup_epochs', [3, 10])
@pytest.mark.parametrize('decay_epochs', [20, 100])
@pytest.mark.parametrize('steps_per_epoch', [10, 20])
@pytest.mark.parametrize('decay_rate', [0.1, 0.5])
@pytest.mark.parametrize('min_lr', [0.00001, 0.00005])
@pytest.mark.parametrize('lr', [0.1, 0.0001])
@pytest.mark.parametrize('sched', ['polynomial_decay', 'warmup_cosine_decay', 'exponential_decay', 'step_decay', 'constant', 'multi_step_decay'])
def test_scheduler(sched, lr, min_lr, decay_rate, steps_per_epoch, decay_epochs, warmup_epochs, milestones):

    if 'warmup' in sched:
        warmup_epochs = warmup_epochs
    else:
        warmup_epochs = 0

    scheduler = create_scheduler(steps_per_epoch, sched, lr=lr, min_lr=min_lr, warmup_epochs=warmup_epochs,
                                 decay_epochs=decay_epochs, decay_rate=decay_rate, milestones=milestones)
    
    # check warmup
    global_step = Tensor(1, ms.int32)
    warmup_epoch = warmup_epochs
    warmup_step = Tensor(warmup_epoch * steps_per_epoch, ms.int32)
    if 'warmup' in sched:
        cur_lr = scheduler(global_step)
        warmup_lr = scheduler(warmup_step)
        if cur_lr > warmup_lr:
            raise ValueError(f'Invalid scheduler: {scheduler}')

    # check decay trend 
    cur_epoch = warmup_epochs + 1
    global_step = Tensor(cur_epoch * steps_per_epoch, ms.int32)

    
    if sched not in ['constant']:
        if sched in ['multi_step_decay']:
            global_step = cur_epoch * steps_per_epoch
        cur_lr = scheduler(global_step)
        if cur_lr > lr:
            raise ValueError(f'Invalid scheduler: {scheduler}')
    else:
        cur_lr = scheduler
        if cur_lr > lr:
            raise ValueError(f'Invalid scheduler: {scheduler}')


    # check value correctness
    cur_epoch = warmup_epochs
    global_step = Tensor(cur_epoch * steps_per_epoch, ms.int32)

    if sched not in ['constant']:
        if sched in ['multi_step_decay']:
            global_step = milestones[3] * steps_per_epoch + 1
            cur_lr = scheduler(global_step)
            x = cur_lr
            y = lr * decay_rate ** 4
            if (abs(x - y) / (0.5 * (x + y))) > 0.001:
                raise ValueError(f'Invalid scheduler: {scheduler}')
        else:
            cur_lr = scheduler(global_step)
            if cur_lr != lr:
                raise ValueError(f'Invalid scheduler: {scheduler}')
    else:
        cur_lr = scheduler
        if cur_lr != lr:
            raise ValueError(f'Invalid scheduler: {scheduler}')

if __name__ == "__main__":
    test_scheduler('multi_step_decay', 0.1, 0.01, 0.1, 10, 5, 2, [1,2,3])
