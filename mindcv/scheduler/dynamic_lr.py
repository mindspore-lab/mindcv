"""Meta learning rate scheduler.

This module implements exactly the same learning rate scheduler as native PyTorch,
see `"torch.optim.lr_scheduler" <https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate>`_.
At present, only `constant_lr`, `linear_lr`, `polynomial_lr`, `exponential_lr`, `step_lr`, `multi_step_lr`,
`cosine_annealing_lr`, `cosine_annealing_warm_restarts_lr`, `one_cycle_lr`, `cyclic_lr` are implemented.
The number, name and usage of the Positional Arguments are exactly the same as those of native PyTorch.

However, due to the constraint of having to explicitly return the learning rate at each step, we have to
introduce additional Keyword Arguments. There are only three Keyword Arguments introduced,
namely `lr`, `steps_per_epoch` and `epochs`, explained as follows:
`lr`: the basic learning rate when creating optim in torch.
`steps_per_epoch`: the number of steps(iterations) of each epoch.
`epochs`: the number of epoch. It and `steps_per_epoch` determine the length of the returned lrs.

In all schedulers, `one_cycle_lr` and `cyclic_lr` only need two Keyword Arguments except `lr`, since
when creating optim in torch, `lr` argument will have no effect if using the two schedulers above.

Since most scheduler in PyTorch are coarse-grained, that is the learning rate is constant within a single epoch.
For non-stepwise scheduler, we introduce several fine-grained variation, that is the learning rate
is also changed within a single epoch. The function name of these variants have the `refined` keyword.
The implemented fine-grained variation are list as follows: `linear_refined_lr`, `polynomial_refined_lr`, etc.

"""
import math
from bisect import bisect_right


def constant_lr(factor, total_iters, *, lr, steps_per_epoch, epochs):
    steps = steps_per_epoch * epochs
    lrs = []
    for i in range(steps):
        epoch_idx = math.floor(i / steps_per_epoch)
        if epoch_idx < total_iters:
            lrs.append(lr * factor)
        else:
            lrs.append(lr)
    return lrs


def linear_lr(start_factor, end_factor, total_iters, *, lr, steps_per_epoch, epochs):
    steps = steps_per_epoch * epochs
    lrs = []
    start_lr = lr * start_factor
    end_lr = lr * end_factor
    for i in range(steps):
        epoch_idx = math.floor(i / steps_per_epoch)
        multiplier = min(epoch_idx, total_iters) / total_iters
        lrs.append(start_lr + multiplier * (end_lr - start_lr))
    return lrs


def linear_refined_lr(start_factor, end_factor, total_iters, *, lr, steps_per_epoch, epochs):
    steps = steps_per_epoch * epochs
    lrs = []
    start_lr = lr * start_factor
    end_lr = lr * end_factor
    for i in range(steps):
        epoch_idx = i / steps_per_epoch
        multiplier = min(epoch_idx, total_iters) / total_iters
        lrs.append(start_lr + multiplier * (end_lr - start_lr))
    return lrs


def polynomial_lr(total_iters, power, *, lr, steps_per_epoch, epochs):
    steps = steps_per_epoch * epochs
    lrs = []
    for i in range(steps):
        epoch_idx = math.floor(i / steps_per_epoch)
        lrs.append(lr * (1 - min(epoch_idx, total_iters) / total_iters) ** power)
    return lrs


def polynomial_refined_lr(total_iters, power, *, lr, steps_per_epoch, epochs):
    steps = steps_per_epoch * epochs
    lrs = []
    for i in range(steps):
        epoch_idx = i / steps_per_epoch
        lrs.append(lr * (1 - min(epoch_idx, total_iters) / total_iters) ** power)
    return lrs


def exponential_lr(gamma, *, lr, steps_per_epoch, epochs):
    steps = steps_per_epoch * epochs
    lrs = []
    for i in range(steps):
        epoch_idx = math.floor(i / steps_per_epoch)
        lrs.append(lr * gamma**epoch_idx)
    return lrs


def exponential_refined_lr(gamma, *, lr, steps_per_epoch, epochs):
    steps = steps_per_epoch * epochs
    lrs = []
    for i in range(steps):
        epoch_idx = i / steps_per_epoch
        lrs.append(lr * gamma**epoch_idx)
    return lrs


def step_lr(step_size, gamma, *, lr, steps_per_epoch, epochs):
    steps = steps_per_epoch * epochs
    lrs = []
    for i in range(steps):
        epoch_idx = math.floor(i / steps_per_epoch)
        lrs.append(lr * gamma ** math.floor(epoch_idx / step_size))
    return lrs


def multi_step_lr(milestones, gamma, *, lr, steps_per_epoch, epochs):
    steps = steps_per_epoch * epochs
    milestones = sorted(milestones)
    lrs = []
    for i in range(steps):
        epoch_idx = math.floor(i / steps_per_epoch)
        lrs.append(lr * gamma ** bisect_right(milestones, epoch_idx))
    return lrs


def cosine_decay_lr(decay_epochs, eta_min, *, eta_max, steps_per_epoch, epochs, num_cycles=1, cycle_decay=1.0):
    """update every epoch"""
    tot_steps = steps_per_epoch * epochs
    lrs = []

    for c in range(num_cycles):
        lr_max = eta_max * (cycle_decay**c)
        delta = 0.5 * (lr_max - eta_min)
        for i in range(steps_per_epoch * decay_epochs):
            t_cur = math.floor(i / steps_per_epoch)
            t_cur = min(t_cur, decay_epochs)
            lr_cur = eta_min + delta * (1.0 + math.cos(math.pi * t_cur / decay_epochs))
            if len(lrs) < tot_steps:
                lrs.append(lr_cur)
            else:
                break

    if epochs > num_cycles * decay_epochs:
        for i in range((epochs - (num_cycles * decay_epochs)) * steps_per_epoch):
            lrs.append(eta_min)

    return lrs


def cosine_decay_refined_lr(decay_epochs, eta_min, *, eta_max, steps_per_epoch, epochs, num_cycles=1, cycle_decay=1.0):
    """update every step"""
    tot_steps = steps_per_epoch * epochs
    lrs = []

    for c in range(num_cycles):
        lr_max = eta_max * (cycle_decay**c)
        delta = 0.5 * (lr_max - eta_min)
        for i in range(steps_per_epoch * decay_epochs):
            t_cur = i / steps_per_epoch
            t_cur = min(t_cur, decay_epochs)
            lr_cur = eta_min + delta * (1.0 + math.cos(math.pi * t_cur / decay_epochs))
            if len(lrs) < tot_steps:
                lrs.append(lr_cur)
            else:
                break

    if epochs > num_cycles * decay_epochs:
        for i in range((epochs - (num_cycles * decay_epochs)) * steps_per_epoch):
            lrs.append(eta_min)

    return lrs


def cosine_annealing_lr(t_max, eta_min, *, eta_max, steps_per_epoch, epochs):
    steps = steps_per_epoch * epochs
    delta = 0.5 * (eta_max - eta_min)
    lrs = []
    for i in range(steps):
        t_cur = math.floor(i / steps_per_epoch)
        lrs.append(eta_min + delta * (1.0 + math.cos(math.pi * t_cur / t_max)))
    return lrs


# fmt: off
def cosine_annealing_warm_restarts_lr(te, tm, eta_min, *, eta_max, steps_per_epoch, epochs):
    delta = 0.5 * (eta_max - eta_min)
    tt = 0
    te_next = te
    lrs = []
    for epoch_idx in range(epochs):
        for batch_idx in range(steps_per_epoch):
            lrs.append(eta_min + delta * (1.0 + math.cos(tt)))
            tt = tt + math.pi / te / steps_per_epoch
            if tt >= math.pi:
                tt = tt - math.pi
        if epoch_idx + 1 == te_next:  # time to restart
            tt = 0                    # by setting to 0 we set lr to lr_max, see above
            te = te * tm              # change the period of restarts
            te_next = te_next + te    # note the next restart's epoch
    return lrs


def one_cycle_lr(
    max_lr: float,
    pct_start: float = 0.3,
    anneal_strategy: str = "cos",
    div_factor: float = 25.0,
    final_div_factor: float = 10000.0,
    three_phase: bool = False,
    *,
    steps_per_epoch: int,
    epochs: int,
):
    """
    OneCycle learning rate scheduler based on
    '"Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates"
    <https://arxiv.org/abs/1708.07120>'

    Args:
        max_lr: Upper learning rate boundaries in the cycle.
        pct_start: The percentage of the number of steps of increasing learning rate
            in the cycle. Default: 0.3.
        anneal_strategy: Define the annealing strategy: "cos" for cosine annealing,
            "linear" for linear annealing. Default: "cos".
        div_factor: Initial learning rate via initial_lr = max_lr / div_factor.
            Default: 25.0.
        final_div_factor: Minimum learning rate at the end via
            min_lr = initial_lr / final_div_factor. Default: 10000.0.
        three_phase: If True, learning rate will be updated by three-phase according to
            "final_div_factor". Otherwise, learning rate will be updated by two-phase.
            Default: False.
        steps_per_epoch: Number of steps per epoch.
        epochs: Number of total epochs.
    """

    def _annealing_cos(start, end, pct):
        cos_out = math.cos(math.pi * pct) + 1
        return end + (start - end) / 2.0 * cos_out

    def _annealing_linear(start, end, pct):
        return (end - start) * pct + start

    initial_lr = max_lr / div_factor
    min_lr = initial_lr / final_div_factor
    steps = steps_per_epoch * epochs
    step_size_up = float(pct_start * steps) - 1
    step_size_down = float(2 * pct_start * steps) - 2
    step_size_end = float(steps) - 1
    if anneal_strategy == "cos":
        anneal_func = _annealing_cos
    elif anneal_strategy == "linear":
        anneal_func = _annealing_linear
    else:
        raise ValueError(f"anneal_strategy must be one of 'cos' or 'linear', but got {anneal_strategy}")
    lrs = []
    for i in range(steps):
        if three_phase:
            if i <= step_size_up:
                lrs.append(anneal_func(initial_lr, max_lr, i / step_size_up))
            elif step_size_up < i <= step_size_down:
                lrs.append(anneal_func(max_lr, initial_lr, (i - step_size_up) / (step_size_down - step_size_up)))
            else:
                lrs.append(anneal_func(initial_lr, min_lr, (i - step_size_down) / (step_size_end - step_size_down)))
        else:
            if i <= step_size_up:
                lrs.append(anneal_func(initial_lr, max_lr, i / step_size_up))
            else:
                lrs.append(anneal_func(max_lr, min_lr, (i - step_size_up) / (step_size_end - step_size_up)))
    return lrs


def cyclic_lr(
    base_lr: float,
    max_lr: float,
    step_size_up: int = 2000,
    step_size_down=None,
    mode: str = "triangular",
    gamma=1.0,
    scale_fn=None,
    scale_mode="cycle",
    *,
    steps_per_epoch: int,
    epochs: int,
):
    """
    Cyclic learning rate scheduler based on
    '"Cyclical Learning Rates for Training Neural Networks" <https://arxiv.org/abs/1708.07120>'

    Args:
        base_lr: Lower learning rate boundaries in each cycle.
        max_lr: Upper learning rate boundaries in each cycle.
        step_size_up: Number of steps in the increasing half in each cycle. Default: 2000.
        step_size_down: Number of steps in the increasing half in each cycle. If step_size_down
            is None, it's set to step_size_up. Default: None.
        div_factor: Initial learning rate via initial_lr = max_lr / div_factor.
            Default: 25.0.
        final_div_factor: Minimum learning rate at the end via
            min_lr = initial_lr / final_div_factor. Default: 10000.0.
        mode: One of {triangular, triangular2, exp_range}. If scale_fn is not None, it's set to
            None. Default: 'triangular'.
        gamma: Constant in 'exp_range' calculating fuction: gamma**(cycle_iterations).
            Default: 1.0
        scale_fn: Custom scaling policy defined by a single argument lambda function. If it's
            not None, 'mode' is ignored. Default: None
        scale_mode: One of {'cycle', 'iterations'}. Determine scale_fn is evaluated on cycle
            number or cycle iterations. Default: 'cycle'
        steps_per_epoch: Number of steps per epoch.
        epochs: Number of total epochs.
    """

    def _triangular_scale_fn(x):
        return 1.0

    def _triangular2_scale_fn(x):
        return 1 / (2.0**(x - 1))

    def _exp_range_scale_fn(x):
        return gamma**x

    steps = steps_per_epoch * epochs
    step_size_up = float(step_size_up)
    step_size_down = float(step_size_down) if step_size_down is not None else step_size_up
    total_size = step_size_up + step_size_down
    step_ratio = step_size_up / total_size
    if scale_fn is None:
        if mode == "triangular":
            scale_fn = _triangular_scale_fn
            scale_mode = "cycle"
        elif mode == "triangular2":
            scale_fn = _triangular2_scale_fn
            scale_mode = "cycle"
        elif mode == "exp_range":
            scale_fn = _exp_range_scale_fn
            scale_mode = "iterations"
    lrs = []
    for i in range(steps):
        cycle = math.floor(1 + i / total_size)
        x = 1.0 + i / total_size - cycle
        if x <= step_ratio:
            scale_factor = x / step_ratio
        else:
            scale_factor = (x - 1) / (step_ratio - 1)
        base_height = (max_lr - base_lr) * scale_factor
        if scale_mode == "cycle":
            lrs.append(base_lr + base_height * scale_fn(cycle))
        else:
            lrs.append(base_lr + base_height * scale_fn(i))
    return lrs


if __name__ == "__main__":
    # Demonstrate how these schedulers work by printing & visualizing the returned list.
    import matplotlib.pyplot as plt
    table = (
        (("constant_lr", constant_lr(0.5, 4, lr=0.05, steps_per_epoch=2, epochs=10),),),
        (("linear_lr", linear_lr(0.5, 1.0, 4, lr=0.05, steps_per_epoch=2, epochs=10)),
         ("linear_refined_lr", linear_refined_lr(0.5, 1.0, 4, lr=0.05, steps_per_epoch=2, epochs=10)),),
        (("polynomial_lr", polynomial_lr(4, 1.0, lr=0.05, steps_per_epoch=2, epochs=10)),
         ("polynomial_refined_lr", polynomial_refined_lr(4, 1.0, lr=0.05, steps_per_epoch=2, epochs=10)),),
        (("exponential_lr", exponential_lr(0.9, lr=0.05, steps_per_epoch=2, epochs=10)),
         ("exponential_refined_lr", exponential_refined_lr(0.9, lr=0.05, steps_per_epoch=2, epochs=10)),),
        (("step_lr", step_lr(3, 0.5, lr=0.05, steps_per_epoch=2, epochs=10)),
         ("multi_step_lr", multi_step_lr([3, 6], 0.5, lr=0.05, steps_per_epoch=2, epochs=10)),),
        (("cosine_decay_lr", cosine_decay_lr(5, 1.0, eta_max=2.0, steps_per_epoch=2, epochs=10)),
         ("cosine_decay_refined_lr", cosine_decay_refined_lr(5, 1.0, eta_max=2.0, steps_per_epoch=2, epochs=10)),),
        (("cosine_annealing_lr", cosine_annealing_lr(5, 0.0, eta_max=1.0, steps_per_epoch=2, epochs=15)),
         (
             "cosine_annealing_warm_restarts_lr",
             cosine_annealing_warm_restarts_lr(5, 2, 0.0, eta_max=1.0, steps_per_epoch=2, epochs=15),
         ),),
        (("one_cycle_lr", one_cycle_lr(2.5, 0.3, "cos", 25.0, 10000.0, False, steps_per_epoch=2, epochs=15)),),
        (("cyclic_lr", cyclic_lr(0.1, 2.5, 5, 5, "triangular", 1.0, None, "cycle", steps_per_epoch=2, epochs=15)),),
    )
    for variants in table:
        n_variants = len(variants)
        fig = plt.figure(figsize=(4, 3 * n_variants))
        for ax_idx, (title, lrs_ms) in enumerate(variants, start=1):
            print(f"name: {title}\nlrs: {lrs_ms}")
            ax = plt.subplot(n_variants, 1, ax_idx)
            ax.plot(lrs_ms, marker="*")
            ax.set_title(title)
            ax.set_xlim(0, len(lrs_ms))  # n_steps
            ax.set_xlabel("step")
            ax.set_ylabel("lr")
        plt.tight_layout()

    # Compare the difference between cosine_annealing_lr and cosine_annealing_warm_restarts_lr.
    plt.figure()
    lrs_ms = cosine_annealing_lr(5, 0.0, eta_max=1.0, steps_per_epoch=10, epochs=35)
    plt.plot(lrs_ms)
    lrs_ms = cosine_annealing_warm_restarts_lr(5, 2, 0.0, eta_max=1.0, steps_per_epoch=10, epochs=35)
    plt.plot(lrs_ms)
    plt.xlabel("step")
    plt.ylabel("lr")
    plt.legend(["cosine_annealing_lr", "cosine_annealing_warm_restarts_lr"], loc="best")
    plt.title("cosine_annealing_lr vs. cosine_annealing_warm_restarts_lr")
    plt.show()
# fmt: on
