import math
import numpy as np
from typing import Optional

import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor

from .layers.conv_norm_act import Conv2dNormActivation
from .layers.squeeze_excite import SqueezeExcite
from .layers.pooling import GlobalAvgPooling
from .utils import load_pretrained, make_divisible
from .registry import register_model

__all__ = [
    'GhostNet',
    'ghostnet_1x',
    'ghostnet_nose_1x'
]

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000,
        'first_conv': '', 'classifier': '',
        **kwargs
    }

default_cfgs = {
    'ghostnet_1x': _cfg(url=''),
    'ghostnet_nose_1x': _cfg(url=''),
}

class mygate(nn.Cell):

    def __init__(self):
        super(mygate, self).__init__()
        self.relu6 = nn.ReLU6()

    def construct(self, x):
        return self.relu6(x + 3.) * 0.16666667

class GhostModule(nn.Cell):
    def __init__(self, num_in: int,
                 num_out: int,
                 norm: Optional[nn.Cell],
                 pad_mode: str = "pad",
                 kernel_size: int = 1,
                 stride: int = 1,
                 ratio: int = 2,
                 dw_size: int = 3,
                 activation: Optional[nn.Cell] = nn.ReLU) -> None:
        super(GhostModule, self).__init__()
        init_channels = math.ceil(num_out / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = Conv2dNormActivation(num_in, init_channels, kernel_size=kernel_size, stride=stride, padding=0,
                                                 pad_mode=pad_mode, groups=1, norm=norm, activation=activation)
        self.cheap_operation = Conv2dNormActivation(init_channels, new_channels, kernel_size=dw_size, stride=stride, padding=dw_size // 2,
                                                    pad_mode=pad_mode, groups=init_channels, norm=norm,
                                                    activation=activation)

    def construct(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        return ops.concat((x1, x2), axis=1)

class GhostBottleneck(nn.Cell):
    def __init__(self, num_in: int,
                 num_mid: int,
                 num_out: int,
                 kernel_size: int,
                 norm: Optional[nn.Cell],
                 pad_mode: str = "pad",
                 stride: int = 1,
                 activation: Optional[nn.Cell] = nn.ReLU,
                 use_se: bool = False) -> None:
        super(GhostBottleneck, self).__init__()
        # Point-wise expansion
        self.num_in = num_in
        self.ghost1 = GhostModule(num_in, num_mid, kernel_size=1, stride=1, pad_mode=pad_mode,
                                  norm=norm, activation=activation)

        # Depth-wise convolution
        self.use_dw = stride > 1
        self.dw = None
        if self.use_dw:
            self.dw = Conv2dNormActivation(num_mid, num_mid, kernel_size=kernel_size, stride=stride,
                                           pad_mode=pad_mode, groups=num_mid, norm=norm, activation=None)

        # Squeeze-and-excitation
        self.use_se = use_se
        if use_se:
            self.se = SqueezeExcite(num_mid, rd_ratio=1./4, rd_divisor=4, act_layer=nn.ReLU(), gate_layer=mygate())

        # Point-wise linear projection
        self.ghost2 = GhostModule(num_mid, num_out, kernel_size=1, stride=1,
                                  pad_mode=pad_mode, norm=norm, activation=None)

        self.down_sample = False
        if num_in != num_out or stride != 1:
            self.down_sample = True

        # shortcut
        self.shortcut = None
        if self.down_sample:
            self.shortcut = nn.SequentialCell([
                Conv2dNormActivation(num_in, num_in, kernel_size=kernel_size, stride=stride,
                                     pad_mode=pad_mode, groups=num_in, norm=norm, activation=None),
                Conv2dNormActivation(num_in, num_out, kernel_size=1, stride=1, padding=0,
                                     pad_mode=pad_mode, groups=1, norm=norm, activation=None),
            ])

    def construct(self, x:Tensor) -> Tensor:
        shortcut = x
        out = self.ghost1(x)
        if self.use_dw:
            out = self.dw(out)
        if self.use_se:
            out = self.se(out)
        out = self.ghost2(out)
        if self.down_sample:
            shortcut = self.shortcut(shortcut)
        out = ops.add(shortcut, out)
        return out


class GhostNet(nn.Cell):
    def __init__(self,
                 model_args,
                 norm: Optional[nn.Cell] = None,
                 num_classes: int = 1000,
                 round_nearest: int = 4,
                 multiplier: float = 1.,
                 final_drop: float = 0.,
                 pad_mode: str = "pad",
                 activation: Optional[nn.Cell] = nn.ReLU) -> None:
        super(GhostNet, self).__init__()
        self.cfgs = model_args['cfg']
        self.inplanes = 16
        layers = []

        # building first layer
        first_conv_in_channel = 3
        first_conv_out_channel = make_divisible(multiplier * self.inplanes, round_nearest)
        if not norm:
            norm = nn.BatchNorm2d

        layers.append(
            Conv2dNormActivation(
                first_conv_in_channel,
                first_conv_out_channel,
                kernel_size=3,
                stride=2,
                pad_mode=pad_mode,
                padding=1,
                norm=norm,
                activation=activation
            )
        )

        # building inverted residual blocks
        for layer_cfg in self.cfgs:
            layers.append(self._make_layer(kernel_size=layer_cfg[0],
                                           exp_ch=make_divisible(multiplier * layer_cfg[1], round_nearest),
                                           out_channel=make_divisible(multiplier * layer_cfg[2], round_nearest),
                                           use_se=layer_cfg[3],
                                           activation=layer_cfg[4],
                                           stride=layer_cfg[5],
                                           norm=norm
                                           )
                          )

        # building last several layers
        output_channel = make_divisible(multiplier * model_args["cls_ch_squeeze"], round_nearest)
        layers.append(
            Conv2dNormActivation(
                make_divisible(multiplier * self.cfgs[-1][2], round_nearest),
                output_channel,
                kernel_size=1,
                stride=1,
                pad_mode=pad_mode,
                groups=1,
                norm=norm,
                activation=activation
            )
        )

        self.features = nn.SequentialCell(layers)

        self.global_pool = GlobalAvgPooling(keep_dims=True)
        self.conv_head = nn.SequentialCell([
            nn.Conv2d(in_channels=output_channel,
                      out_channels=model_args['cls_ch_expand'],
                      kernel_size=1,
                      padding=0,
                      stride=1,
                      has_bias=True,
                      pad_mode='pad'),
            nn.ReLU(),
        ])
        self.final_drop = final_drop
        if self.final_drop > 0:
            self.dropout = nn.Dropout(self.final_drop)
        self.classifier = nn.Dense(model_args['cls_ch_expand'], num_classes, has_bias=True)

        self._initialize_weights()

    def forward_features(self, x: Tensor) -> Tensor:
        x = self.features(x)
        return self.global_pool(x)

    def forward_head(self, x: Tensor) -> Tensor:
        x = self.conv_head(x)
        x = ops.flatten(x)
        if self.final_drop > 0:
            x = self.dropout(x)
        return self.classifier(x)

    def construct(self, x: Tensor) -> Tensor:
        x = self.forward_features(x)
        return self.forward_head(x)

    def _make_layer(self, kernel_size: int,
                    exp_ch: int,
                    out_channel: int,
                    use_se: bool,
                    activation: Optional[nn.Cell],
                    norm: Optional[nn.Cell],
                    stride: int = 1):
        layer = GhostBottleneck(self.inplanes, exp_ch, out_channel, kernel_size, stride=stride,
                                use_se=use_se, activation=activation, norm=norm)
        self.inplanes = out_channel

        return layer

    def _initialize_weights(self):
        self.init_parameters_data()
        for _, m in self.cells_and_names():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.set_data(Tensor(np.random.normal(0, np.sqrt(2. / n), m.weight.data.shape).astype("float32")))
                if m.bias is not None:
                    m.bias.set_data(Tensor(np.zeros(m.bias.data.shape, dtype="float32")))
            elif isinstance(m, nn.BatchNorm2d):
                m.gamma.set_data(Tensor(np.ones(m.gamma.data.shape, dtype="float32")))
                m.beta.set_data(Tensor(np.zeros(m.beta.data.shape, dtype="float32")))
            elif isinstance(m, nn.Dense):
                m.weight.set_data(Tensor(np.random.normal(0, 0.01, m.weight.data.shape).astype("float32")))
                if m.bias is not None:
                    m.bias.set_data(Tensor(np.zeros(m.bias.data.shape, dtype="float32")))


model_cfgs = {
    "1x": {
        "cfg": [
            # k, exp, c,  se,     nl,  s,
            # stage1
            [3, 16, 16, False, nn.ReLU, 1],
            # stage2
            [3, 48, 24, False, nn.ReLU, 2],
            [3, 72, 24, False, nn.ReLU, 1],
            # stage3
            [5, 72, 40, True, nn.ReLU, 2],
            [5, 120, 40, True, nn.ReLU, 1],
            # stage4
            [3, 240, 80, False, nn.ReLU, 2],
            [3, 200, 80, False, nn.ReLU, 1],
            [3, 184, 80, False, nn.ReLU, 1],
            [3, 184, 80, False, nn.ReLU, 1],
            [3, 480, 112, True, nn.ReLU, 1],
            [3, 672, 112, True, nn.ReLU, 1],
            # stage5
            [5, 672, 160, True, nn.ReLU, 2],
            [5, 960, 160, False, nn.ReLU, 1],
            [5, 960, 160, True, nn.ReLU, 1],
            [5, 960, 160, False, nn.ReLU, 1],
            [5, 960, 160, True, nn.ReLU, 1]],
        "cls_ch_squeeze": 960,
        "cls_ch_expand": 1280,
    },

    "nose_1x": {
        "cfg": [
            # k, exp, c,  se,     nl,  s,
            # stage1
            [3, 16, 16, False, nn.ReLU, 1],
            # stage2
            [3, 48, 24, False, nn.ReLU, 2],
            [3, 72, 24, False, nn.ReLU, 1],
            # stage3
            [5, 72, 40, False, nn.ReLU, 2],
            [5, 120, 40, False, nn.ReLU, 1],
            # stage4
            [3, 240, 80, False, nn.ReLU, 2],
            [3, 200, 80, False, nn.ReLU, 1],
            [3, 184, 80, False, nn.ReLU, 1],
            [3, 184, 80, False, nn.ReLU, 1],
            [3, 480, 112, False, nn.ReLU, 1],
            [3, 672, 112, False, nn.ReLU, 1],
            # stage5
            [5, 672, 160, False, nn.ReLU, 2],
            [5, 960, 160, False, nn.ReLU, 1],
            [5, 960, 160, False, nn.ReLU, 1],
            [5, 960, 160, False, nn.ReLU, 1],
            [5, 960, 160, False, nn.ReLU, 1]],
        "cls_ch_squeeze": 960,
        "cls_ch_expand": 1280,
    }
}


@register_model
def ghostnet_1x(pretrained: bool = False,
                num_classes: int = 1000,
                in_channels: int = 3,
                multiplier: float = 1.,
                final_drop: float = 0.8,
                **kwargs):
    model_args = default_cfgs['ghostnet_1x']
    model = GhostNet(norm=nn.BatchNorm2d,
                     model_args=model_cfgs['1x'],
                     num_classes=num_classes,
                     multiplier=multiplier,
                     final_drop=final_drop,
                     **kwargs)

    if pretrained:
        load_pretrained(model, model_args, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def ghostnet_nose_1x(pretrained: bool = False,
                     num_classes: int = 1000,
                     in_channels: int = 3,
                     multiplier: float = 1.,
                     final_drop: float = 0.8,
                     **kwargs):
    model_args = default_cfgs['ghostnet_nose_1x']
    model = GhostNet(model_args=model_cfgs['nose_1x'],
                     num_classes=num_classes,
                     multiplier=multiplier,
                     final_drop=final_drop,
                     **kwargs)

    if pretrained:
        load_pretrained(model, model_args, num_classes=num_classes, in_channels=in_channels)

    return model
