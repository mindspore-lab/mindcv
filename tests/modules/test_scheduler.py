import sys
sys.path.append('.')

import pytest

import mindspore as ms
from mindspore import Tensor
from mindcv.scheduler import create_scheduler


@pytest.mark.parametrize('decay_epochs', [5, 10])
@pytest.mark.parametrize('steps_per_epoch', [100, 200])
@pytest.mark.parametrize('decay_rate', [0.1, 0.5])
@pytest.mark.parametrize('min_lr', [0.00001, 0.00005])
@pytest.mark.parametrize('lr', [0.1, 0.001])
@pytest.mark.parametrize('sched', ['polynomial_decay', 'exponential_decay', 'step_decay'])
def test_scheduler_poly_exp_step(sched, lr, min_lr, decay_rate, steps_per_epoch, decay_epochs):

    warmup_epochs = 0

    scheduler = create_scheduler(steps_per_epoch, sched, lr=lr, min_lr=min_lr, warmup_epochs=warmup_epochs,
                                 decay_epochs=decay_epochs, decay_rate=decay_rate)
    
    # check warmup
    # check decay trend 
    cur_epoch = warmup_epochs + 1
    global_step = Tensor(cur_epoch * steps_per_epoch, ms.int32)

    
    if sched in ['step_decay']:
        cur_epoch = decay_epochs
        global_step = Tensor(cur_epoch * steps_per_epoch, ms.int32)
        cur_lr = scheduler(global_step)
    else:
        cur_epoch = warmup_epochs + 1
        global_step = Tensor(cur_epoch * steps_per_epoch, ms.int32)
        cur_lr = scheduler(global_step)
    assert cur_lr < lr, 'lr does NOT decrease'


    # check value correctness
    cur_epoch = warmup_epochs
    global_step = Tensor(cur_epoch * steps_per_epoch, ms.int32)
    cur_lr = scheduler(global_step)
    cur_lr = Tensor.asnumpy(cur_lr)
    assert abs(cur_lr - lr) < 1e-6, 'Incorrect lr'


@pytest.mark.parametrize('decay_epochs', [10])
@pytest.mark.parametrize('steps_per_epoch', [200])
@pytest.mark.parametrize('decay_rate', [0.5])
@pytest.mark.parametrize('min_lr', [0.00001])
@pytest.mark.parametrize('lr', [0.1, 0.001])
@pytest.mark.parametrize('sched', ['constant'])
def test_scheduler_cons(sched, lr, min_lr, decay_rate, steps_per_epoch, decay_epochs):

    scheduler = create_scheduler(steps_per_epoch, sched, lr=lr, min_lr=min_lr,
                                 decay_epochs=decay_epochs, decay_rate=decay_rate)

    cur_lr = scheduler
    assert cur_lr == lr, 'lr is NOT constant'


@pytest.mark.parametrize('milestones', [[1, 3, 5, 7], [2, 4, 6, 8]])
@pytest.mark.parametrize('steps_per_epoch', [100, 200])
@pytest.mark.parametrize('decay_rate', [0.1, 0.5])
@pytest.mark.parametrize('lr', [0.1, 0.001])
@pytest.mark.parametrize('sched', ['multi_step_decay'])
def test_scheduler_multi_step(sched, lr, decay_rate, steps_per_epoch, milestones):

    warmup_epochs = 0

    scheduler = create_scheduler(steps_per_epoch, sched, lr=lr, warmup_epochs=warmup_epochs,
                                 decay_rate=decay_rate, milestones=milestones)

    # check decay trend
    cur_epoch = warmup_epochs + 2
    global_step = cur_epoch * steps_per_epoch +1
    cur_lr = scheduler(global_step)
    assert cur_lr < lr, 'lr does NOT decrease'

    # check value correctness

    global_step = milestones[3] * steps_per_epoch + 1
    cur_lr = scheduler(global_step)
    x = cur_lr
    y = lr * decay_rate ** 4
    assert (abs(x - y) / (0.5 * (x + y))) < 0.001, 'Incorrect lr'



@pytest.mark.parametrize('warmup_epochs', [3, 10])
@pytest.mark.parametrize('decay_epochs', [20, 50])
@pytest.mark.parametrize('steps_per_epoch', [100, 200])
@pytest.mark.parametrize('min_lr', [0.00001, 0.00005])
@pytest.mark.parametrize('lr', [0.1, 0.001])
@pytest.mark.parametrize('sched', ['warmup_cosine_decay'])
def test_scheduler_warm_cos(sched, lr, min_lr, steps_per_epoch, decay_epochs, warmup_epochs):

    warmup_epochs = warmup_epochs

    scheduler = create_scheduler(steps_per_epoch, sched, lr=lr, min_lr=min_lr, warmup_epochs=warmup_epochs,
                                 decay_epochs=decay_epochs)

    # check warmup
    global_step = Tensor(1, ms.int32)
    warmup_epoch = warmup_epochs
    warmup_step = Tensor(warmup_epoch * steps_per_epoch, ms.int32)

    cur_lr = scheduler(global_step)
    warmup_lr = scheduler(warmup_step)
    assert cur_lr < warmup_lr, "lr warmup is NOT incorrect"

    # check decay trend
    cur_epoch = warmup_epochs + 1
    global_step = Tensor(cur_epoch * steps_per_epoch, ms.int32)
    cur_lr = scheduler(global_step)
    assert cur_lr < lr, 'lr does NOT decrease'

    # check value correctness
    cur_epoch = warmup_epochs
    global_step = Tensor(cur_epoch * steps_per_epoch, ms.int32)

    cur_lr = scheduler(global_step)
    assert cur_lr == lr, 'Incorrect lr'



