""" auto mixed precision related functions """
from mindspore import dtype as mstype
from mindspore import nn
from mindspore.ops import functional as F

# from mindspore.amp import LossScaler  # this line of code leads to “get rank id error” in modelarts
try:
    from mindspore import jit_class
except ImportError:
    from mindspore import ms_class as jit_class


@jit_class
class LossScaler:
    r"""
    Loss scaler abstract class when using mixed precision.

    Derived class needs to implement all of its methods. During training, `scale` and `unscale` is used
    to scale and unscale the loss value and gradients to avoid overflow, `adjust` is used to update the
    loss scale value.

    Note:
        This is an experimental interface that is subject to change or deletion.
    """

    def scale(self, inputs):
        """
        Scaling inputs by `scale_value`.

        Args:
            inputs (Union(Tensor, tuple(Tensor))): the input loss value or gradients.
        """
        raise NotImplementedError

    def unscale(self, inputs):
        """
        Unscaling inputs by `scale_value`.

        Args:
            inputs (Union(Tensor, tuple(Tensor))): the input loss value or gradients.
        """
        raise NotImplementedError

    def adjust(self, grads_finite):
        """
        Adjust the `scale_value` dependent on whether grads are finite.

        Args:
            grads_finite (Tensor): a scalar bool Tensor indicating whether the grads are finite.
        """
        raise NotImplementedError


class NoLossScaler(LossScaler):
    """
    No LossScaler
    """

    def scale(self, inputs):
        return inputs

    def unscale(self, inputs):
        return inputs

    def adjust(self, grads_finite):
        return True


AMP_WHITE_LIST = (
    nn.Dense,
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.Conv1dTranspose,
    nn.Conv2dTranspose,
    nn.Conv3dTranspose,
)

AMP_BLACK_LIST = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.LayerNorm,
)


class _OutputTo16(nn.Cell):
    "Wrap cell for amp. Cast network output back to float16"

    def __init__(self, op):
        super(_OutputTo16, self).__init__(auto_prefix=False)
        self._op = op

    def construct(self, x):
        return F.cast(self._op(x), mstype.float16)


class _OutputTo32(nn.Cell):
    "Wrap cell for amp. Cast network output back to float32"

    def __init__(self, op):
        super(_OutputTo32, self).__init__(auto_prefix=False)
        self._op = op

    def construct(self, x):
        return F.cast(self._op(x), mstype.float32)


def _auto_white_list(network, white_list=None):
    """process the white list of network."""
    if white_list is None:
        white_list = AMP_WHITE_LIST
    cells = network.name_cells()
    change = False
    for name in cells:
        subcell = cells[name]
        if subcell == network:
            continue
        elif isinstance(subcell, white_list):
            network._cells[name] = _OutputTo32(subcell.to_float(mstype.float16))
            change = True
        else:
            _auto_white_list(subcell, white_list)
    if isinstance(network, nn.SequentialCell) and change:
        network.cell_list = list(network.cells())


def _auto_black_list(network, black_list=None):
    """process the black list of network."""
    if black_list is None:
        black_list = AMP_BLACK_LIST
    network.to_float(mstype.float16)
    cells = network.name_cells()
    change = False
    for name in cells:
        subcell = cells[name]
        if subcell == network:
            continue
        elif isinstance(subcell, black_list):
            network._cells[name] = _OutputTo16(subcell.to_float(mstype.float32))
            change = True
        else:
            _auto_black_list(subcell, black_list)
    if isinstance(network, nn.SequentialCell) and change:
        network.cell_list = list(network.cells())


def auto_mixed_precision(network, amp_level="O0", amp_cast_list=None):
    """
    auto mixed precision function.
    Args:
        network (Cell): Definition of the network.
        amp_level (str): Supports ["O0", "O1", "O2", "O3"]. Default: "O0".

            - "O0": Do not change.
            - "O1": Cast the operators in white_list to float16, the remaining operators are kept in float32.
            - "O2": Cast network to float16, keep operators in black_list run in float32,
            - "O3": Cast network to float16.
        amp_cast_list: At the cell level, customize the list to cast ops to FP16.

    Raises:
        ValueError: If amp level is not supported.
    """
    if amp_cast_list is not None:
        amp_cast_list = eval(amp_cast_list)
        if isinstance(amp_cast_list, list):
            amp_cast_list = tuple(amp_cast_list)

    if amp_level == "O0":
        pass
    elif amp_level == "O1":
        _auto_white_list(network, amp_cast_list)
    elif amp_level == "O2":
        _auto_black_list(network, amp_cast_list)
    elif amp_level == "O3":
        network.to_float(mstype.float16)
    else:
        raise ValueError("The amp level {} is not supported".format(amp_level))
