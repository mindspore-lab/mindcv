import sys
sys.path.append('.')

import pytest
import numpy as np

import mindspore as ms
from mindspore import Tensor
from mindcv.scheduler import create_scheduler
from mindcv.scheduler import dynamic_lr


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


def test_scheduler_dynamic():
    # constant_lr
    lrs_manually = [0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025,
                    0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
    lrs_ms = dynamic_lr.constant_lr(0.5, 4, lr=0.05, steps_per_epoch=2, epochs=10)
    assert np.allclose(lrs_ms, lrs_manually)

    # linear_lr
    lrs_manually = [0.025, 0.025, 0.03125, 0.03125, 0.0375, 0.0375, 0.04375, 0.04375,
                    0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
    lrs_ms = dynamic_lr.linear_lr(0.5, 1.0, 4, lr=0.05, steps_per_epoch=2, epochs=10)
    assert np.allclose(lrs_ms, lrs_manually)

    # linear_refined_lr
    lrs_manually = [0.025, 0.028125, 0.03125, 0.034375, 0.0375, 0.040625, 0.04375, 0.046875,
                    0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
    lrs_ms = dynamic_lr.linear_refined_lr(0.5, 1.0, 4, lr=0.05, steps_per_epoch=2, epochs=10)
    assert np.allclose(lrs_ms, lrs_manually)

    # polynomial_lr
    lrs_manually = [0.05, 0.05, 0.0375, 0.0375, 0.025, 0.025, 0.0125, 0.0125,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    lrs_ms = dynamic_lr.polynomial_lr(4, 1.0, lr=0.05, steps_per_epoch=2, epochs=10)
    assert np.allclose(lrs_ms, lrs_manually)

    # polynomial_refined_lr
    lrs_manually = [0.05, 0.04375, 0.0375, 0.03125, 0.025, 0.01875, 0.0125, 0.00625,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    lrs_ms = dynamic_lr.polynomial_refined_lr(4, 1.0, lr=0.05, steps_per_epoch=2, epochs=10)
    assert np.allclose(lrs_ms, lrs_manually)

    # exponential_lr
    lrs_manually = [0.05, 0.05, 0.045, 0.045, 0.0405, 0.0405,
                    0.03645, 0.03645, 0.032805, 0.032805, 0.0295245, 0.0295245,
                    0.02657205, 0.02657205, 0.023914845, 0.023914845,
                    0.0215233605, 0.0215233605, 0.01937102445, 0.01937102445]
    lrs_ms = dynamic_lr.exponential_lr(0.9, lr=0.05, steps_per_epoch=2, epochs=10)
    assert np.allclose(lrs_ms, lrs_manually)

    # exponential_refined_lr
    lrs_manually = [0.05, 0.047434164902525694, 0.045, 0.042690748412273126, 0.0405, 0.03842167357104581,
                    0.03645, 0.03457950621394123, 0.032805, 0.031121555592547107, 0.0295245, 0.0280094000332924,
                    0.02657205, 0.02520846002996316, 0.023914845, 0.022687614026966844,
                    0.0215233605, 0.02041885262427016, 0.01937102445, 0.018376967361843143]
    lrs_ms = dynamic_lr.exponential_refined_lr(0.9, lr=0.05, steps_per_epoch=2, epochs=10)
    assert np.allclose(lrs_ms, lrs_manually)

    # step_lr
    lrs_manually = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
                    0.025, 0.025, 0.025, 0.025, 0.025, 0.025,
                    0.0125, 0.0125, 0.0125, 0.0125, 0.0125, 0.0125, 0.00625, 0.00625]
    lrs_ms = dynamic_lr.step_lr(3, 0.5, lr=0.05, steps_per_epoch=2, epochs=10)
    assert np.allclose(lrs_ms, lrs_manually)

    # multi_step_lr
    lrs_manually = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
                    0.025, 0.025, 0.025, 0.025, 0.025, 0.025,
                    0.0125, 0.0125, 0.0125, 0.0125, 0.0125, 0.0125, 0.0125, 0.0125]
    lrs_ms = dynamic_lr.multi_step_lr([3, 6], 0.5, lr=0.05, steps_per_epoch=2, epochs=10)
    assert np.allclose(lrs_ms, lrs_manually)

    # cosine_annealing_lr
    lrs_manually = [1.0, 1.0, 0.9045084971874737, 0.9045084971874737, 0.6545084971874737, 0.6545084971874737,
                    0.34549150281252633, 0.34549150281252633, 0.09549150281252633, 0.09549150281252633,
                    0.0, 0.0, 0.09549150281252622, 0.09549150281252622, 0.3454915028125262, 0.3454915028125262,
                    0.6545084971874736, 0.6545084971874736, 0.9045084971874737, 0.9045084971874737,
                    1.0, 1.0, 0.9045084971874741, 0.9045084971874741, 0.6545084971874738, 0.6545084971874738,
                    0.34549150281252644, 0.34549150281252644, 0.09549150281252639, 0.09549150281252639]
    lrs_ms = dynamic_lr.cosine_annealing_lr(5, 0.0, eta_max=1.0, steps_per_epoch=2, epochs=15)
    assert np.allclose(lrs_ms, lrs_manually)

    # cosine_annealing_warm_restarts_lr
    lrs_manually = [1.0, 0.9755282581475768, 0.9045084971874737, 0.7938926261462366, 0.6545084971874737, 0.5,
                    0.34549150281252633, 0.2061073738537635, 0.09549150281252633, 0.024471741852423234,
                    1.0, 0.9938441702975689, 0.9755282581475768, 0.9455032620941839, 0.9045084971874737,
                    0.8535533905932737, 0.7938926261462366, 0.7269952498697734, 0.6545084971874737, 0.5782172325201155,
                    0.5, 0.4217827674798846, 0.34549150281252633, 0.2730047501302266, 0.2061073738537635,
                    0.14644660940672627, 0.09549150281252633, 0.054496737905816106, 0.024471741852423234,
                    0.00615582970243117]
    lrs_ms = dynamic_lr.cosine_annealing_warm_restarts_lr(5, 2, 0.0, eta_max=1.0, steps_per_epoch=2, epochs=15)
    assert np.allclose(lrs_ms, lrs_manually)
