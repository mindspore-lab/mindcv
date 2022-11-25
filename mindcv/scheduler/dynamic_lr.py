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
        multiplier = min(i, total_iters * steps_per_epoch) / total_iters / steps_per_epoch
        lrs.append(start_lr + multiplier * (end_lr - start_lr))
    return lrs


def multi_step_lr(milestones, gamma, *, lr, steps_per_epoch, epochs):
    steps = steps_per_epoch * epochs
    milestones = sorted(milestones)
    lrs = []
    for i in range(steps):
        epoch_idx = math.floor(i / steps_per_epoch)
        lrs.append(lr * gamma ** bisect_right(milestones, epoch_idx))
    return lrs


def cosine_annealing_lr(t_max, eta_min, *, eta_max, steps_per_epoch, epochs):
    steps = steps_per_epoch * epochs
    delta = 0.5 * (eta_max - eta_min)
    lrs = []
    for i in range(steps):
        t_cur = math.floor(i / steps_per_epoch)
        lrs.append(eta_min + delta * (1.0 + math.cos(math.pi * t_cur / t_max)))
    return lrs


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


if __name__ == '__main__':
    # Demonstrate how these schedulers work by printing returned list.
    print(constant_lr(0.5, 4, lr=0.05, steps_per_epoch=2, epochs=10))
    print(linear_lr(0.5, 1.0, 4, lr=0.05, steps_per_epoch=2, epochs=10))
    print(linear_refined_lr(0.5, 1.0, 4, lr=0.05, steps_per_epoch=2, epochs=10))
    print(multi_step_lr([3, 6], 0.5, lr=0.05, steps_per_epoch=2, epochs=10))
    print(cosine_annealing_lr(5, 0.0, eta_max=1.0, steps_per_epoch=2, epochs=15))
    print(cosine_annealing_warm_restarts_lr(5, 2, 0.0, eta_max=1.0, steps_per_epoch=2, epochs=15))

    # Demonstrate how these schedulers work by visualizing the returned list.
    import matplotlib.pyplot as plt
    fig = plt.figure()
    for ax_idx, (title, lrs_ms) in enumerate([
        ("constant_lr", constant_lr(0.5, 4, lr=0.05, steps_per_epoch=2, epochs=10)),
        ("multi_step_lr", multi_step_lr([3, 6], 0.5, lr=0.05, steps_per_epoch=2, epochs=10)),
        ("linear_lr", linear_lr(0.5, 1.0, 4, lr=0.05, steps_per_epoch=2, epochs=10)),
        ("linear_refined_lr", linear_refined_lr(0.5, 1.0, 4, lr=0.05, steps_per_epoch=2, epochs=10)),
    ], start=1):
        ax = plt.subplot(2, 2, ax_idx)
        ax.plot(lrs_ms, marker="*")
        ax.set_title(title)
        ax.set_xlim(0, 20)
        ax.set_xlabel("step")
        ax.set_ylabel("lr")
    fig.suptitle("lr=0.05, steps_per_epoch=2, epochs=10")
    plt.tight_layout()

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
