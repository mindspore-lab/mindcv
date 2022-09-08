#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""optimizer factory"""

import mindspore.nn as nn
from typing import Optional


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
        **kwargs):

    opt = opt.lower()

    weight_decay = weight_decay
    if weight_decay and filter_bias_and_bn:
        params = init_group_params(params, weight_decay)

    opt_args = dict(**kwargs)
    # if lr is not None:
    #    opt_args.setdefault('lr', lr)

    # non-adaptive: SGD, momentum, and nesterov 
    if opt == 'sgd' or opt == 'nesterov':
        optimizer = nn.SGD(params=params,
                           learning_rate=lr,
                           momentum=momentum,
                           weight_decay=weight_decay,
                           nesterov=nesterov,
                           loss_scale=loss_scale,
                           **opt_args
                           )
    elif opt == 'momentum':
        optimizer = nn.Momentum(params=params,
                                learning_rate=lr,
                                momentum=momentum,
                                weight_decay=weight_decay,
                                use_nesterov=nesterov,
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
        # TODO: the mindspore implementation seems different from the official AdamW 
        optimizer = nn.AdamWeightDecay(params=params,
                                       learning_rate=lr,
                                       weight_decay=weight_decay,
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
