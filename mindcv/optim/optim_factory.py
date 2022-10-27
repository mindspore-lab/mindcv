''' optim factory '''
import os
from typing import Optional
from mindspore import nn
from mindspore import load_checkpoint, load_param_into_net
from .adan import Adan
from .adamw import AdamW
from .nadam import NAdam

__all__ = ["create_optimizer"]


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
        checkpoint_path: str = '',
        **kwargs):
    r"""Creates optimizer by name.

    Args:
        params: network parameters.
        opt: optimizer name like 'sgd', 'nesterov', 'momentum', 'adam', 'adamw', 'rmsprop', 'adagrad', 'lamb'.
            Adam is the default choise for convolution-based networks.
            AdamW is recommended for ViT-based networks. Default: 'adam'.
        lr: learning rate: float or lr scheduler. Fixed and dynamic learning rate are supported. Default: 1e-3.
        weight_decay: weight decay factor. Default: 0.
        momentum: momentum if the optimizer supports. Default: 0.9.
        nesterov: Whether to use Nesterov Accelerated Gradient (NAG) algorithm to update the gradients. Default: False.
        filter_bias_and_bn: whether to filter batch norm paramters and bias from weight decay.
            If True, weight decay will not apply on BN parameters and bias in Conv or Dense layers. Default: True.
        loss_scale: A floating point value for the loss scale, which must be larger than 0.0. Default: 1.0.

    Returns:
        Optimizer object
    """

    opt = opt.lower()

    if weight_decay and filter_bias_and_bn:
        params = init_group_params(params, weight_decay)

    opt_args = dict(**kwargs)
    # if lr is not None:
    #    opt_args.setdefault('lr', lr)

    # non-adaptive: SGD, momentum, and nesterov
    if opt == 'sgd':
        # note: nn.Momentum may perform better if momentum > 0.
        optimizer = nn.SGD(params=params,
                           learning_rate=lr,
                           momentum=momentum,
                           weight_decay=weight_decay,
                           nesterov=nesterov,
                           loss_scale=loss_scale,
                           **opt_args
                           )
    elif opt in ['momentum', 'nesterov']:
        optimizer = nn.Momentum(params=params,
                           learning_rate=lr,
                           momentum=momentum,
                           weight_decay=weight_decay,
                           use_nesterov=nesterov,
                           loss_scale=loss_scale,
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
        assert loss_scale == 1.0, 'Loss scaler is not supported by Lamb optimizer'
        optimizer = nn.Lamb(params=params,
                            learning_rate=lr,
                            weight_decay=weight_decay,
                            **opt_args)
    else:
        raise ValueError(f'Invalid optimizer: {opt}')

    if os.path.exists(checkpoint_path):
        param_dict = load_checkpoint(checkpoint_path)
        load_param_into_net(optimizer, param_dict)

    return optimizer
