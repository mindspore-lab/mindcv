from enum import Enum
from typing import Union

import mindspore


class Format(str, Enum):
    NCHW = 'NCHW'
    NHWC = 'NHWC'
    NCL = 'NCL'
    NLC = 'NLC'


FormatT = Union[str, Format]


def nchw_to(x: mindspore.Tensor, fmt: Format):
    if fmt == Format.NHWC:
        x = x.permute(0, 2, 3, 1)
    elif fmt == Format.NLC:
        x = x.flatten(start_dim=2).transpose((0, 2, 1))
    elif fmt == Format.NCL:
        x = x.flatten(start_dim=2)
    return x


def nhwc_to(x: mindspore.Tensor, fmt: Format):
    if fmt == Format.NCHW:
        x = x.permute(0, 3, 1, 2)
    elif fmt == Format.NLC:
        x = x.flatten(start_dim=1, end_dim=2)
    elif fmt == Format.NCL:
        x = x.flatten(start_dim=1, end_dim=2).transpose((0, 2, 1))
    return x
