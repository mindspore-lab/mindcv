import inspect

import mindspore as ms
from mindspore import nn, ops

__all__ = [
    "Dropout",
    "Interpolate",
    "Split",
]


class Dropout(nn.Dropout):
    def __init__(self, p=0.5, dtype=ms.float32):
        sig = inspect.signature(super().__init__)
        if "keep_prob" in sig.parameters and "p" not in sig.parameters:
            super().__init__(keep_prob=1.0-p, dtype=dtype)
        elif "p" in sig.parameters:
            super().__init__(p=p, dtype=dtype)
        else:
            raise NotImplementedError(
                f"'keep_prob' or 'p' must be the parameter of `mindspore.nn.Dropout`, but got signature of it: {sig}."
            )


class Interpolate(nn.Cell):
    def __init__(self, scale_factor=None, mode="nearest", align_corners=None, recompute_scale_factor=None):
        super().__init__()
        sig = inspect.signature(ops.interpolate)
        if "sizes" in sig.parameters:
            if scale_factor is None and recompute_scale_factor is None:
                self.kwargs = dict(
                    roi=None,
                    scales=None,
                    coordinate_transformation_mode="align_corners" if align_corners is True else "half_pixel",
                    mode=mode,
                )
                self.size_name = "sizes"
            else:
                raise NotImplementedError(
                    "'scale_factor' and 'recompute_scale_factor' do not supported in mindspore 1.x!"
                )
        elif "size" in sig.parameters:
            self.kwargs = dict(
                scale_factor=scale_factor,
                mode=mode,
                align_corners=align_corners,
                recompute_scale_factor=recompute_scale_factor,
            )
            self.size_name = "size"
        else:
            raise NotImplementedError(
                f"'sizes' or 'size' must be the parameter of `mindspore.ops.interpolate`, "
                f"but got signature of it: {sig}."
            )

    def construct(self, x, size=None):
        if self.size_name == "sizes":
            return ops.interpolate(x, sizes=size, **self.kwargs)
        elif self.size_name == "size":
            return ops.interpolate(x, size=size, **self.kwargs)
        else:
            return None


class Split(nn.Cell):
    """Splits the tensor into chunks.
    In order to ensure that your code can run on different versions of mindspore,
    you need to pass in two sets of redundant parameters.
    """
    def __init__(self, split_size_or_sections, output_num, axis=0):
        super().__init__()
        sig = inspect.signature(ops.split)
        if "output_num" in sig.parameters:
            self.kwargs = dict(axis=axis, output_num=output_num)
        elif "split_size_or_sections" in sig.parameters:
            self.kwargs = dict(split_size_or_sections=split_size_or_sections, axis=axis)
        else:
            raise NotImplementedError(
                f"'output_num' or 'split_size_or_sections' must be the parameter of `mindspore.ops.split`, "
                f"but got signature of it: {sig}."
            )

    def construct(self, x):
        return ops.split(x, **self.kwargs)
