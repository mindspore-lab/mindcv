""" optim factory """
import os
from functools import partial
from typing import Optional

from mindspore import load_checkpoint, load_param_into_net, nn

from .adamw import AdamW
from .adan import Adan
from .lion import Lion
from .nadam import NAdam

__all__ = ["create_optimizer", "create_pretrain_optimizer", "create_finetune_optimizer"]


def init_group_params(params, weight_decay):
    decay_params = []
    no_decay_params = []

    for param in params:
        if "beta" not in param.name and "gamma" not in param.name and "bias" not in param.name:
            decay_params.append(param)
        else:
            no_decay_params.append(param)
    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params},
        {"order_params": params},
    ]


def create_optimizer(
    params,
    opt: str = "adam",
    lr: Optional[float] = 1e-3,
    weight_decay: float = 0,
    momentum: float = 0.9,
    nesterov: bool = False,
    filter_bias_and_bn: bool = True,
    loss_scale: float = 1.0,
    schedule_decay: float = 4e-3,
    checkpoint_path: str = "",
    eps: float = 1e-10,
    **kwargs,
):
    r"""Creates optimizer by name.

    Args:
        params: network parameters. Union[list[Parameter],list[dict]], which must be the list of parameters
            or list of dicts. When the list element is a dictionary, the key of the dictionary can be
            "params", "lr", "weight_decay","grad_centralization" and "order_params".
        opt: wrapped optimizer. You could choose like 'sgd', 'nesterov', 'momentum', 'adam', 'adamw', 'lion',
            'rmsprop', 'adagrad', 'lamb'. 'adam' is the default choose for convolution-based networks.
            'adamw' is recommended for ViT-based networks. Default: 'adam'.
        lr: learning rate: float or lr scheduler. Fixed and dynamic learning rate are supported. Default: 1e-3.
        weight_decay: weight decay factor. It should be noted that weight decay can be a constant value or a Cell.
            It is a Cell only when dynamic weight decay is applied. Dynamic weight decay is similar to
            dynamic learning rate, users need to customize a weight decay schedule only with global step as input,
            and during training, the optimizer calls the instance of WeightDecaySchedule to get the weight decay value
            of current step. Default: 0.
        momentum: momentum if the optimizer supports. Default: 0.9.
        nesterov: Whether to use Nesterov Accelerated Gradient (NAG) algorithm to update the gradients. Default: False.
        filter_bias_and_bn: whether to filter batch norm parameters and bias from weight decay.
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

    optimizer = get_optimizer(
        params, opt_args, opt, lr, weight_decay, momentum, nesterov, loss_scale, schedule_decay, checkpoint_path, eps
    )

    return optimizer


def get_pretrain_param_groups(model, weight_decay, skip, skip_keywords):
    """get pretrain param groups"""
    has_decay, has_decay_name = [], []
    no_decay, no_decay_name = [], []

    for param in model.trainable_params():
        if (
            len(param.shape) == 1
            or param.name.endswith(".bias")
            or (param.name in skip)
            or check_keywords_in_name(param.name, skip_keywords)
        ):
            no_decay.append(param)
            no_decay_name.append(param.name)
        else:
            has_decay.append(param)
            has_decay_name.append(param.name)

    return [
        {"params": has_decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
        {"order_params": model.trainable_params()},
    ]


def create_pretrain_optimizer(
    model,
    opt: str = "adam",
    lr: Optional[float] = 1e-3,
    weight_decay: float = 0,
    momentum: float = 0.9,
    nesterov: bool = False,
    filter_bias_and_bn: bool = True,
    loss_scale: float = 1.0,
    schedule_decay: float = 4e-3,
    checkpoint_path: str = "",
    eps: float = 1e-10,
    **kwargs,
):
    """build pretrain optimizer"""

    opt = opt.lower()

    skip = {}
    skip_keywords = {}
    if hasattr(model, "no_weight_decay"):
        skip = model.no_weight_decay()
    if hasattr(model, "no_weight_decay_keywords"):
        skip_keywords = model.no_weight_decay_keywords()

    params = get_pretrain_param_groups(model, weight_decay, skip, skip_keywords)

    opt_args = dict(**kwargs)
    # if lr is not None:
    #    opt_args.setdefault('lr', lr)

    optimizer = get_optimizer(
        params, opt_args, opt, lr, weight_decay, momentum, nesterov, loss_scale, schedule_decay, checkpoint_path, eps
    )

    return optimizer


def get_vit_layer(name, num_layers):
    if name in ("cls_token", "mask_token", "pos_embed"):
        return 0
    elif name.startswith("patch_embed"):
        return 0
    elif name.startswith("rel_pos_bias"):
        return num_layers - 1
    elif name.startswith("blocks"):
        layer_id = int(name.split(".")[1])
        return layer_id + 1
    else:
        return num_layers - 1


def get_swin_layer(name, num_layers, depths):
    if name in ("mask_token",):
        return 0
    elif name.startswith("patch_embed"):
        return 0
    elif name.startswith("layers"):
        layer_id = int(name.split(".")[1])
        block_id = name.split(".")[3]
        if block_id == "reduction" or block_id == "norm":
            return sum(depths[: layer_id + 1])
        layer_id = sum(depths[:layer_id]) + int(block_id)
        return layer_id + 1
    else:
        return num_layers - 1


def get_finetune_param_groups(
    model,
    lr,
    weight_decay,
    get_layer_func,
    scales,
    skip,
    skip_keywords,
):
    parameter_group_names = {}
    parameter_group_vars = {}

    for param in model.trainable_params():
        if (
            len(param.shape) == 1
            or param.name.endswith(".bias")
            or (param.name in skip)
            or check_keywords_in_name(param.name, skip_keywords)
        ):
            group_name = "no_decay"
            this_weight_decay = 0.0
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if get_layer_func is not None:
            layer_id = get_layer_func(param.name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if scales is not None:
                scale = scales[layer_id]
            else:
                scale = 1.0

            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr": [learning_rate * scale for learning_rate in lr],
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr": [learning_rate * scale for learning_rate in lr],
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(param.name)

    return list(parameter_group_vars.values())


def create_finetune_optimizer(
    model,
    opt: str = "adam",
    lr: Optional[float] = 1e-3,
    weight_decay: float = 0,
    momentum: float = 0.9,
    nesterov: bool = False,
    filter_bias_and_bn: bool = True,
    loss_scale: float = 1.0,
    schedule_decay: float = 4e-3,
    checkpoint_path: str = "",
    eps: float = 1e-10,
    scale: float = 0.75,
    **kwargs,
):
    if hasattr(model, "get_depths"):
        depths = model.get_depths()
        num_layers = model.get_num_layers()
        get_layer_func = partial(get_swin_layer, num_layers=num_layers + 2, depths=depths)
    elif hasattr(model, "get_num_layers"):
        num_layers = model.get_num_layers()
        get_layer_func = partial(get_vit_layer, num_layers=num_layers + 2)
    else:
        raise NotImplementedError()

    scales = list(scale**i for i in reversed(range(num_layers + 2)))

    skip = {}
    skip_keywords = {}
    if hasattr(model, "no_weight_decay"):
        skip = model.no_weight_decay()
    if hasattr(model, "no_weight_decay_keywords"):
        skip_keywords = model.no_weight_decay_keywords()

    params = get_finetune_param_groups(model, lr, weight_decay, get_layer_func, scales, skip, skip_keywords)

    opt_args = dict(**kwargs)
    # if lr is not None:
    #    opt_args.setdefault('lr', lr)

    optimizer = get_optimizer(
        params, opt_args, opt, lr, weight_decay, momentum, nesterov, loss_scale, schedule_decay, checkpoint_path, eps
    )

    return optimizer


def get_optimizer(
    params,
    opt_args,
    opt: str = "adam",
    lr: Optional[float] = 1e-3,
    weight_decay: float = 0,
    momentum: float = 0.9,
    nesterov: bool = False,
    loss_scale: float = 1.0,
    schedule_decay: float = 4e-3,
    checkpoint_path: str = "",
    eps: float = 1e-10,
):
    # non-adaptive: SGD, momentum, and nesterov
    if opt == "sgd":
        # note: nn.Momentum may perform better if momentum > 0.
        optimizer = nn.SGD(
            params=params,
            learning_rate=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            loss_scale=loss_scale,
            **opt_args,
        )
    elif opt in ["momentum", "nesterov"]:
        optimizer = nn.Momentum(
            params=params,
            learning_rate=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            use_nesterov=nesterov,
            loss_scale=loss_scale,
        )
    # adaptive
    elif opt == "adam":
        optimizer = nn.Adam(
            params=params,
            learning_rate=lr,
            weight_decay=weight_decay,
            loss_scale=loss_scale,
            use_nesterov=nesterov,
            **opt_args,
        )
    elif opt == "adamw":
        optimizer = AdamW(
            params=params,
            learning_rate=lr,
            weight_decay=weight_decay,
            loss_scale=loss_scale,
            **opt_args,
        )
    elif opt == "lion":
        optimizer = Lion(
            params=params,
            learning_rate=lr,
            weight_decay=weight_decay,
            loss_scale=loss_scale,
            **opt_args,
        )
    elif opt == "nadam":
        optimizer = NAdam(
            params=params,
            learning_rate=lr,
            weight_decay=weight_decay,
            loss_scale=loss_scale,
            schedule_decay=schedule_decay,
            **opt_args,
        )
    elif opt == "adan":
        optimizer = Adan(
            params=params,
            learning_rate=lr,
            weight_decay=weight_decay,
            loss_scale=loss_scale,
            **opt_args,
        )
    elif opt == "rmsprop":
        optimizer = nn.RMSProp(
            params=params,
            learning_rate=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            loss_scale=loss_scale,
            epsilon=eps,
            **opt_args,
        )
    elif opt == "adagrad":
        optimizer = nn.Adagrad(
            params=params,
            learning_rate=lr,
            weight_decay=weight_decay,
            loss_scale=loss_scale,
            **opt_args,
        )
    elif opt == "lamb":
        assert loss_scale == 1.0, "Loss scaler is not supported by Lamb optimizer"
        optimizer = nn.Lamb(
            params=params,
            learning_rate=lr,
            weight_decay=weight_decay,
            **opt_args,
        )
    else:
        raise ValueError(f"Invalid optimizer: {opt}")

    if os.path.exists(checkpoint_path):
        param_dict = load_checkpoint(checkpoint_path)
        load_param_into_net(optimizer, param_dict)

    return optimizer


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin
