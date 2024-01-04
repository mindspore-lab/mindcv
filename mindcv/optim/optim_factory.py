""" optim factory """
import collections
import logging
import os
import re
from collections import defaultdict
from itertools import chain, islice
from typing import Any, Callable, Dict, Iterator, Optional, Tuple, Union

from mindspore import load_checkpoint, load_param_into_net, nn

from .adamw import AdamW
from .adan import Adan
from .lion import Lion
from .nadam import NAdam

__all__ = ["create_optimizer"]

_logger = logging.getLogger(__name__)


def init_group_params(params, weight_decay, weight_decay_filter, no_weight_decay):
    if weight_decay_filter == "disable":
        return [
            {"params": params, "weight_decay": weight_decay},
            {"order_params": params},
        ]

    decay_params = []
    no_decay_params = []
    no_weight_decay = set(no_weight_decay)
    for param in params:
        if "beta" in param.name or "gamma" in param.name or "bias" in param.name or param.name in no_weight_decay:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
        {"order_params": params},
    ]


def param_groups_layer_decay(
    model: nn.Cell,
    lr: Optional[float] = 1e-3,
    weight_decay: float = 0.05,
    no_weight_decay_list: Tuple[str] = (),
    layer_decay: float = 0.75,
):
    """
    Parameter groups for layer-wise lr decay & weight decay
    """
    no_weight_decay_list = set(no_weight_decay_list)
    param_group_names = {}  # NOTE for debugging
    param_groups = {}
    if hasattr(model, "group_matcher"):
        layer_map = group_with_matcher(model.trainable_params(), model.group_matcher(coarse=False), reverse=True)
    else:
        layer_map = _layer_map(model)

    num_layers = max(layer_map.values()) + 1
    layer_max = num_layers - 1
    layer_scales = list(layer_decay ** (layer_max - i) for i in range(num_layers))

    for name, param in model.parameters_and_names():
        if not param.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        if param.ndim == 1 or name in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.0
        else:
            g_decay = "decay"
            this_decay = weight_decay

        layer_id = layer_map.get(name, layer_max)
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_groups:
            this_scale = layer_scales[layer_id]
            param_group_names[group_name] = {
                "lr": [learning_rate * this_scale for learning_rate in lr],
                "weight_decay": this_decay,
                "param_names": [],
            }
            param_groups[group_name] = {
                "lr": [learning_rate * this_scale for learning_rate in lr],
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["param_names"].append(name)
        param_groups[group_name]["params"].append(param)

    return list(param_groups.values())


MATCH_PREV_GROUP = (99999,)


def group_with_matcher(
    named_objects: Iterator[Tuple[str, Any]], group_matcher: Union[Dict, Callable], reverse: bool = False
):
    if isinstance(group_matcher, dict):
        # dictionary matcher contains a dict of raw-string regex expr that must be compiled
        compiled = []
        for group_ordinal, (_, mspec) in enumerate(group_matcher.items()):
            if mspec is None:
                continue
            # map all matching specifications into 3-tuple (compiled re, prefix, suffix)
            if isinstance(mspec, (tuple, list)):
                # multi-entry match specifications require each sub-spec to be a 2-tuple (re, suffix)
                for sspec in mspec:
                    compiled += [(re.compile(sspec[0]), (group_ordinal,), sspec[1])]
            else:
                compiled += [(re.compile(mspec), (group_ordinal,), None)]
        group_matcher = compiled

    def _get_grouping(name):
        if isinstance(group_matcher, (list, tuple)):
            for match_fn, prefix, suffix in group_matcher:
                r = match_fn.match(name)
                if r:
                    parts = (prefix, r.groups(), suffix)
                    # map all tuple elem to int for numeric sort, filter out None entries
                    return tuple(map(float, chain.from_iterable(filter(None, parts))))
            return (float("inf"),)  # un-matched layers (neck, head) mapped to largest ordinal
        else:
            ord = group_matcher(name)
            if not isinstance(ord, collections.abc.Iterable):
                return (ord,)
            return tuple(ord)

    grouping = defaultdict(list)
    for param in named_objects:
        grouping[_get_grouping(param.name)].append(param.name)
    # remap to integers
    layer_id_to_param = defaultdict(list)
    lid = -1
    for k in sorted(filter(lambda x: x is not None, grouping.keys())):
        if lid < 0 or k[-1] != MATCH_PREV_GROUP[0]:
            lid += 1
        layer_id_to_param[lid].extend(grouping[k])

    if reverse:
        # output reverse mapping
        param_to_layer_id = {}
        for lid, lm in layer_id_to_param.items():
            for n in lm:
                param_to_layer_id[n] = lid
        return param_to_layer_id

    return layer_id_to_param


def _group(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def _layer_map(model, layers_per_group=12, num_groups=None):
    def _in_head(n, hp):
        if not hp:
            return True
        elif isinstance(hp, (tuple, list)):
            return any([n.startswith(hpi) for hpi in hp])
        else:
            return n.startswith(hp)

    # attention: need to add pretrained_cfg attr to model
    head_prefix = getattr(model, "pretrained_cfg", {}).get("classifier", None)
    names_trunk = []
    names_head = []
    for n, _ in model.parameters_and_names():
        names_head.append(n) if _in_head(n, head_prefix) else names_trunk.append(n)

    # group non-head layers
    num_trunk_layers = len(names_trunk)
    if num_groups is not None:
        layers_per_group = -(num_trunk_layers // -num_groups)
    names_trunk = list(_group(names_trunk, layers_per_group))
    num_trunk_groups = len(names_trunk)
    layer_map = {n: i for i, l in enumerate(names_trunk) for n in l}
    layer_map.update({n: num_trunk_groups for n in names_head})
    return layer_map


def create_optimizer(
    model_or_params,
    opt: str = "adam",
    lr: Optional[float] = 1e-3,
    weight_decay: float = 0,
    momentum: float = 0.9,
    nesterov: bool = False,
    weight_decay_filter: str = "disable",
    layer_decay: Optional[float] = None,
    loss_scale: float = 1.0,
    schedule_decay: float = 4e-3,
    checkpoint_path: str = "",
    eps: float = 1e-10,
    **kwargs,
):
    r"""Creates optimizer by name.

    Args:
        model_or_params: network or network parameters. Union[list[Parameter],list[dict], nn.Cell], which must be
            the list of parameters or list of dicts or nn.Cell. When the list element is a dictionary, the key of
            the dictionary can be "params", "lr", "weight_decay","grad_centralization" and "order_params".
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
        weight_decay_filter: filters to filter parameters from weight_decay.
            - "disable": No parameters to filter.
            - "auto": We do not apply weight decay filtering to any parameters. However, MindSpore currently
                    automatically filters the parameters of Norm layer from weight decay.
            - "norm_and_bias": Filter the paramters of Norm layer and Bias from weight decay.
        layer_decay: for apply layer-wise learning rate decay.
        loss_scale: A floating point value for the loss scale, which must be larger than 0.0. Default: 1.0.

    Returns:
        Optimizer object
    """

    no_weight_decay = {}
    if isinstance(model_or_params, nn.Cell):
        # a model was passed in, extract parameters and add weight decays to appropriate layers
        if hasattr(model_or_params, "no_weight_decay"):
            no_weight_decay = model_or_params.no_weight_decay()
        params = model_or_params.trainable_params()

    else:
        params = model_or_params

    if weight_decay_filter == "auto":
        _logger.warning(
            "You are using AUTO weight decay filter, which means the weight decay filter isn't explicitly pass in "
            "when creating an mindspore.nn.Optimizer instance. "
            "NOTE: mindspore.nn.Optimizer will filter Norm parmas from weight decay. "
        )
    elif layer_decay is not None and isinstance(model_or_params, nn.Cell):
        params = param_groups_layer_decay(
            model_or_params,
            lr=lr,
            weight_decay=weight_decay,
            layer_decay=layer_decay,
            no_weight_decay_list=no_weight_decay,
        )
        weight_decay = 0.0
    elif weight_decay_filter == "disable" or "norm_and_bias":
        params = init_group_params(params, weight_decay, weight_decay_filter, no_weight_decay)
        weight_decay = 0.0
    else:
        raise ValueError(
            f"weight decay filter only support ['disable', 'auto', 'norm_and_bias'], but got{weight_decay_filter}."
        )

    opt = opt.lower()
    opt_args = dict(**kwargs)
    # if lr is not None:
    #    opt_args.setdefault('lr', lr)

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
