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
# ============================================================================

import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.numpy import empty


def drop_path(x: Tensor,
              drop_prob: float = 0.,
              training: bool = False,
              scale_by_keep: bool = True) -> Tensor:
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = ops.bernoulli(empty(shape), p=keep_prob)
    if keep_prob > 0. and scale_by_keep:
        random_tensor = ops.div(random_tensor, keep_prob)
    return x * random_tensor


class DropPath(nn.Cell):

    def __init__(self,
                 drop_prob: float = 0.,
                 scale_by_keep: bool = True) -> None:
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def construct(self, x: Tensor) -> Tensor:
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)
