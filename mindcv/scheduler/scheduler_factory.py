from mindspore.nn import ExponentialDecayLR

from .warmup_cosine_decay_lr import WarmupCosineDecayLR


def create_scheduler(
        steps_per_epoch: int,
        scheduler: str = 'constant',
        lr: float = 0.01,
        min_lr: float = 1e-6,
        warmup_epochs: int = 3,
        decay_epochs: int = 10,
        decay_rate: float = 0.9,
):
    r"""
    Args:
        steps_per_epoch: number of steps per epoch
        scheduler: 'constant' constant lr, 'warmup_consine_decay', 'step_decay', 'exponential_decay'  
        lr: learning rate value. 
        min_lr: lower lr bound for cyclic/cosine schedulers that hit 0 (1e-5)
        warmup_epochs: epochs to warmup LR, if scheduler supports 
        decay_epochs: epochs to decay LR to min_lr for cyclic schedulers. decay LR by decay_rate every `decay_epochs` for exponential scheduler and step LR scheduler
        decay_rate: LR decay rate (default: 0.9) 
    
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
    elif scheduler == 'step_decay':
        decay_steps = decay_epochs * steps_per_epoch
        # decay LR by decay_rate every `decay_steps`
        lr_scheduler = ExponentialDecayLR(lr,
                                          decay_rate,
                                          decay_steps,
                                          is_stair=True
                                          )
    elif scheduler == 'const':
        lr_scheduler = lr
    else:
        raise ValueError(f'Invalid scheduler: {scheduler}')

    return lr_scheduler
