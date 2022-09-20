import mindspore.nn as nn
from typing import Optional
from .adan import Adan
from .adamw import AdamW
from .nadam import NAdam


def init_group_params(params, weight_decay):
    decay_params = []
    no_decay_params = []

    for param in params:
        if 'beta' not in param.name and 'gamma' not in param.name and 'bias' not in param.name:
            decay_params.append(param)
        else:
            no_decay_params.append(param)
    return [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params},
        {'order_params': params}
    ]


def create_optimizer(
        params,
        opt: str = 'adam',
        lr: Optional[float] = 1e-3,
        weight_decay: float = 0,
        momentum: float = 0.9,
        nesterov: bool = False,
        filter_bias_and_bn: bool = True,
        loss_scale: float = 1.0,
        schedule_decay: float = 4e-3,
        **kwargs):
    '''
    opt: optimizer name, default 'adam' for covolution-based networks. 'AdamW' is recommended for ViT-based networks.
    '''

    opt = opt.lower()

    weight_decay = weight_decay
    if weight_decay and filter_bias_and_bn:
        params = init_group_params(params, weight_decay)

    opt_args = dict(**kwargs)
    # if lr is not None:
    #    opt_args.setdefault('lr', lr)

    # non-adaptive: SGD, momentum, and nesterov 
    if opt == 'sgd' or opt == 'nesterov' or opt=='momentum':
        optimizer = nn.SGD(params=params,
                           learning_rate=lr,
                           momentum=momentum,
                           weight_decay=weight_decay,
                           nesterov=nesterov,
                           loss_scale=loss_scale,
                           **opt_args
                           )
    
    # adaptive 
    elif opt == 'adam':
        optimizer = nn.Adam(params=params,
                            learning_rate=lr,
                            weight_decay=weight_decay,
                            loss_scale=loss_scale,
                            use_nesterov=nesterov,
                            **opt_args)
    elif opt == 'adamw':
        optimizer = AdamW(params=params,
                            learning_rate=lr,
                            weight_decay=weight_decay,
                            loss_scale=loss_scale,
                            **opt_args)
    elif opt == 'nadam':
        optimizer = NAdam(params=params,
                            learning_rate=lr,
                            weight_decay=weight_decay,
                            loss_scale=loss_scale,
                            schedule_decay=schedule_decay,
                            **opt_args)
    elif opt == 'adan':
        optimizer = Adan(params=params,
                            learning_rate=lr,
                            weight_decay=weight_decay,
                            loss_scale=loss_scale,
                            **opt_args)
    elif opt == 'rmsprop':
        optimizer = nn.RMSProp(params=params,
                               learning_rate=lr,
                               momentum=momentum,
                               weight_decay=weight_decay,
                               loss_scale=loss_scale,
                               **opt_args
                               )
    elif opt == 'adagrad':
        optimizer = nn.Adagrad(params=params,
                               learning_rate=lr,
                               weight_decay=weight_decay,
                               loss_scale=loss_scale,
                               **opt_args)
    elif opt == 'lamb':
        optimizer = nn.Lamb(params=params,
                            learning_rate=lr,
                            weight_decay=weight_decay,
                            **opt_args)
    else:
        raise ValueError(f'Invalid optimizer: {opt}')

    return optimizer
