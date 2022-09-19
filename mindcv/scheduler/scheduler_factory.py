from mindspore.nn import ExponentialDecayLR, PolynomialDecayLR

from .warmup_cosine_decay_lr import WarmupCosineDecayLR
from .multi_step_decay_lr import MultiStepDecayLR


def create_scheduler(
        steps_per_epoch: int,
        scheduler: str = 'constant',
        lr: float = 0.01,
        min_lr: float = 1e-6,
        warmup_epochs: int = 3,
        decay_epochs: int = 10,
        decay_rate: float = 0.9,
        milestones: list = []
):
    r"""
    Args:
        steps_per_epoch: number of steps per epoch
        scheduler: 'constant' for constant lr, 'warmup_consine_decay', 'step_decay', 'exponential_decay', 'polynomial_decay', 'multi_step_decay'
        lr: learning rate value. 
        min_lr: lower lr bound for cyclic/cosine/polynomial schedulers
        warmup_epochs: epochs to warmup LR, if scheduler supports 
        decay_epochs: epochs to decay LR to min_lr for cyclic and polynomial schedulers. decay LR by a factor of decay_rate every `decay_epochs` for exponential scheduler and step LR scheduler
        decay_rate: LR decay rate (default: 0.9)
        milestones: list of epoch milestones for multi_step_decay scheduler. Must be increasing.

    
    Returns: 
        Cell object for computing LR with input of current global steps 

    """
    if scheduler == 'warmup_cosine_decay':
        lr_scheduler = WarmupCosineDecayLR(min_lr=min_lr,
                                           max_lr=lr,
                                           warmup_epochs=warmup_epochs,
                                           decay_epochs=decay_epochs,
                                           steps_per_epoch=steps_per_epoch
                                           )
    elif scheduler == 'exponential_decay':
        decay_steps = decay_epochs * steps_per_epoch
        lr_scheduler = ExponentialDecayLR(lr,
                                          decay_rate,
                                          decay_steps,
                                          is_stair=False
                                          )
    elif scheduler == 'polynomial_decay':
        decay_steps = decay_epochs * steps_per_epoch
        lr_scheduler = PolynomialDecayLR(lr, 
                                         min_lr, # end_learning_rate 
                                         decay_steps, 
                                         power=decay_rate, # overload decay_rate as polynomial power 
                                         update_decay_steps=False)

    elif scheduler == 'step_decay':
        decay_steps = decay_epochs * steps_per_epoch
        # decay LR by decay_rate every `decay_steps`
        lr_scheduler = ExponentialDecayLR(lr,
                                          decay_rate,
                                          decay_steps,
                                          is_stair=True
                                          )

    elif scheduler == 'multi_step_decay':
        decay_step_indices = [epoch * steps_per_epoch for epoch in milestones]
        # decay LR by decay_rate once the step reaches `decay_step_indices`
        lr_scheduler = MultiStepDecayLR(lr,
                                        decay_rate,
                                        decay_step_indices
                                        )

    elif scheduler == 'constant':
        lr_scheduler = lr
    else:
        raise ValueError(f'Invalid scheduler: {scheduler}')

    return lr_scheduler
