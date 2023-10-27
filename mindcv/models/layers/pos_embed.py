"""positional embedding"""
import math
from typing import List, Optional, Tuple

import numpy as np

import mindspore as ms
from mindspore import Parameter, Tensor, nn, ops

from .compatibility import Interpolate


def resample_abs_pos_embed(
        posemb,
        new_size: List[int],
        old_size: Optional[List[int]] = None,
        num_prefix_tokens: int = 1,
        interpolation: str = 'nearest',
):
    # sort out sizes, assume square if old size not provided
    num_pos_tokens = posemb.shape[1]
    num_new_tokens = new_size[0] * new_size[1] + num_prefix_tokens

    if num_new_tokens == num_pos_tokens and new_size[0] == new_size[1]:
        return posemb

    if old_size is None:
        hw = int(math.sqrt(num_pos_tokens - num_prefix_tokens))
        old_size = hw, hw

    if num_prefix_tokens:
        posemb_prefix, posemb = posemb[:, :num_prefix_tokens], posemb[:, num_prefix_tokens:]
    else:
        posemb_prefix, posemb = None, posemb

    # do the interpolation
    embed_dim = posemb.shape[-1]
    orig_dtype = posemb.dtype
    posemb = posemb.reshape(1, old_size[0], old_size[1], -1).permute(0, 3, 1, 2)
    interpolate = Interpolate(mode=interpolation, align_corners=True)
    posemb = interpolate(posemb, size=new_size)
    posemb = posemb.permute(0, 2, 3, 1).reshape(1, -1, embed_dim)
    posemb = posemb.astype(orig_dtype)

    # add back extra (class, etc) prefix tokens
    if posemb_prefix is not None:
        posemb = ops.concatcat((posemb_prefix, posemb), axis=1)

    return posemb


class RelativePositionBiasWithCLS(nn.Cell):
    def __init__(
        self,
        window_size: Tuple[int],
        num_heads: int
    ):
        super(RelativePositionBiasWithCLS, self).__init__()
        self.window_size = window_size
        self.num_tokens = window_size[0] * window_size[1]

        num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        # 3: cls to token, token to cls, cls to cls
        self.relative_position_bias_table = Parameter(
            Tensor(np.zeros((num_relative_distance, num_heads)), dtype=ms.float16)
        )
        coords_h = np.arange(window_size[0]).reshape(window_size[0], 1).repeat(window_size[1], 1).reshape(1, -1)
        coords_w = np.arange(window_size[1]).reshape(1, window_size[1]).repeat(window_size[0], 0).reshape(1, -1)
        coords_flatten = np.concatenate([coords_h, coords_w], axis=0)  # [2, Wh * Ww]

        relative_coords = coords_flatten[:, :, np.newaxis] - coords_flatten[:, np.newaxis, :]  # [2, Wh * Ww, Wh * Ww]
        relative_coords = relative_coords.transpose(1, 2, 0)  # [Wh * Ww, Wh * Ww, 2]
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[0] - 1

        relative_position_index = np.zeros((self.num_tokens + 1, self.num_tokens + 1),
                                           dtype=relative_coords.dtype)  # [Wh * Ww + 1, Wh * Ww + 1]
        relative_position_index[1:, 1:] = relative_coords.sum(-1)
        relative_position_index[0, 0:] = num_relative_distance - 3
        relative_position_index[0:, 0] = num_relative_distance - 2
        relative_position_index[0, 0] = num_relative_distance - 1
        relative_position_index = Tensor(relative_position_index.reshape(-1))

        self.one_hot = nn.OneHot(axis=-1, depth=num_relative_distance, dtype=ms.float16)
        self.relative_position_index = Parameter(self.one_hot(relative_position_index), requires_grad=False)

    def construct(self):
        out = ops.matmul(self.relative_position_index, self.relative_position_bias_table)
        out = ops.reshape(out, (self.num_tokens + 1, self.num_tokens + 1, -1))
        out = ops.transpose(out, (2, 0, 1))
        out = ops.expand_dims(out, 0)
        return out
