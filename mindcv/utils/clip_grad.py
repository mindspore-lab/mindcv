"""
Gradient of clipping
"""

from typing import Union, List, Tuple

from mindspore import Tensor, ops
import mindspore.numpy as msnp


def clip_grad_norm(params_gradient: List[Tensor],
                   max_norm: Union[float, int],
                   norm_type: Union[float, int, str] = 2) -> Tuple[Tensor, List[Tensor]]:
    r"""Clips gradient norm of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector.

    Args:
        params_gradient (List[Tensor]): a list of Tensors or a
            single Tensor that will have gradients normalized.
        max_norm (float or int): max norm of the gradients.
        norm_type (int or str): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector) and gradient after clipping.
    """
    if isinstance(params_gradient, Tensor):
        params_gradient = [params_gradient]
    max_norm = float(max_norm)
    norm_type = float(norm_type)

    if norm_type == float('inf'):
        total_norm = max(g.abs().max() for g in params_gradient)
    else:
        norms = ops.stack([ops.norm(g, axis=list(range(g.ndim)), p=norm_type) for g in params_gradient])
        total_norm = ops.norm(norms, axis=list(range(norms.ndim)), p=norm_type)

    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef_clamped = ops.clip_by_value(clip_coef, clip_value_max=Tensor(1.0))

    params_gradient = [ops.mul(g, clip_coef_clamped) for g in params_gradient]

    return total_norm, params_gradient


def clip_grad_value(params_gradient: List[Tensor],
                    clip_value: Union[float, int]) -> List[Tensor]:
    r"""Clips gradient of parameters at specified value.

    Args:
        params_gradient (List[Tensor]): a list of Tensors or a
            single Tensor that will have gradients normalized.
        clip_value (float or int): maximum allowed value of the gradients.

    Returns:
        gradient after clipping.
    """
    if isinstance(params_gradient, Tensor):
        params_gradient = [params_gradient]

    clip_value = Tensor(float(clip_value))

    params_gradient = [ops.clip_by_value(g, clip_value_min=-clip_value, clip_value_max=clip_value) for g in
                       params_gradient]

    return params_gradient


def unitwise_norm(x, norm_type=2):
    if x.ndim == 1:
        return ops.norm(x, axis=0, p=norm_type)
    else:
        return ops.norm(x, axis=list(range(1, x.ndim)), p=norm_type, keep_dims=True)


def adaptive_clip_grad(parameters: List[Tensor],
                       params_gradient: List[Tensor],
                       clip_factor: float = 0.01,
                       eps: float = 1e-3,
                       norm_type: int = 2) -> List[Tensor]:
    r"""Clips gradient of parameters at specified value.


        Args:
            parameters (List[Tensor]): Trainable parameters.
            params_gradient (List[Tensor]): a list of Tensors or a
                single Tensor that will have gradients normalized.
            clip_factor (float): Clipping factor.
            eps (float): The minimum value.
            norm_type (int): type of the used p-norm.

        Returns:
            gradient after clipping.
        """
    if isinstance(params_gradient, Tensor):
        parameters = [params_gradient]

    new_grads = []
    for p, g in zip(parameters, params_gradient):
        max_norm = ops.mul(ops.clip_by_value(unitwise_norm(p, norm_type=norm_type), clip_value_min=eps), clip_factor)
        grad_norm = unitwise_norm(g, norm_type=norm_type)
        clipped_grad = g * (max_norm / ops.clip_by_value(grad_norm, clip_value_min=1e-6))
        new_grads.append(msnp.where(grad_norm < max_norm, g, clipped_grad))
        return new_grads
