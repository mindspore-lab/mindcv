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
"""Conv2d + BN + ACT"""

from typing import Optional

from mindspore import nn


class Conv2dNormActivation(nn.Cell):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 pad_mode: str = "pad",
                 padding: Optional[int] = None,
                 dilation: int = 1,
                 groups: int = 1,
                 norm: Optional[nn.Cell] = nn.BatchNorm2d,
                 activation: Optional[nn.Cell] = nn.ReLU,
                 has_bias: Optional[bool] = None,
                 **kwargs) -> None:
        super(Conv2dNormActivation, self).__init__()

        if pad_mode == "pad":
            if padding is None:
                padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
        else:
            padding = 0

        if has_bias is None:
            has_bias = norm is None

        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                pad_mode=pad_mode,
                padding=padding,
                dilation=dilation,
                group=groups,
                has_bias=has_bias,
                **kwargs)
        ]

        if norm:
            layers.append(norm(out_channels))
        if activation:
            layers.append(activation())

        self.features = nn.SequentialCell(layers)

    def construct(self, x):
        output = self.features(x)
        return output
