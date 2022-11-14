import mindspore.common.initializer as init
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
from mindspore import dtype as mstype
from mindspore.nn import Cell
from .registry import register_model
from .utils import load_pretrained

__all__ = [
    'regnetx200mf',
    'regnetx400mf',
    'regnetx600mf',
    'regnetx800mf',
    'regnetx1_6gf',
    'regnetx3_2gf',
    'regnetx4_0gf',
    'regnetx6_4gf',
    'regnetx8_0gf',
    'regnetx12gf',
    'regnetx16gf',
    'regnetx32gf',
    'regnety200mf',
    'regnety400mf',
    'regnety600mf',
    'regnety800mf',
    'regnety1_6gf',
    'regnety3_2gf',
    'regnety4_0gf',
    'regnety6_4gf',
    'regnety8_0gf',
    'regnety12gf',
    'regnety16gf',
    'regnety32gf'
]


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000,
        'first_conv': '', 'classifier': 'classifier',
        **kwargs
    }


default_cfgs = {
    'regnetx200mf': _cfg(url=''),
    'regnetx400mf': _cfg(url=''),
    'regnetx600mf': _cfg(url=''),
    'regnetx800mf': _cfg(url=''),
    'regnetx1_6gf': _cfg(url=''),
    'regnetx3_2gf': _cfg(url=''),
    'regnetx4_0gf': _cfg(url=''),
    'regnetx6_4gf': _cfg(url=''),
    'regnetx8_0gf': _cfg(url=''),
    'regnetx12gf': _cfg(url=''),
    'regnetx16gf': _cfg(url=''),
    'regnetx32gf': _cfg(url=''),
    'regnety200mf': _cfg(url=''),
    'regnety400mf': _cfg(url=''),
    'regnety600mf': _cfg(url=''),
    'regnety800mf': _cfg(url=''),
    'regnety1_6gf': _cfg(url=''),
    'regnety3_2gf': _cfg(url=''),
    'regnety4_0gf': _cfg(url=''),
    'regnety6_4gf': _cfg(url=''),
    'regnety8_0gf': _cfg(url=''),
    'regnety12gf': _cfg(url=''),
    'regnety16gf': _cfg(url=''),
    'regnety32gf': _cfg(url=''),
}


def conv2d(w_in, w_out, k, *, stride=1, groups=1, bias=False):
    """Helper for building a conv2d layer."""
    assert k % 2 == 1, "Only odd size kernels supported to avoid padding issues."
    s, p, g, b = stride, (k - 1) // 2, groups, bias
    return nn.Conv2d(w_in, w_out, k, stride=s, padding=p, group=g, has_bias=b, pad_mode='pad',
                     weight_init=init.initializer(init.HeNormal(mode='fan_out', nonlinearity='relu'),
                                                  [w_out, w_in // g, k, k], mstype.float32))


def norm2d(w_in, eps, mom):
    """Helper for building a norm2d layer."""
    return nn.BatchNorm2d(num_features=w_in, eps=eps, momentum=mom)


def pool2d(_w_in, k, *, stride=1):
    """Helper for building a pool2d layer."""
    assert k % 2 == 1, "Only odd size kernels supported to avoid padding issues."
    padding = (k - 1) // 2
    pad2d = nn.Pad(((0, 0), (0, 0), (padding, padding), (padding, padding)), mode="CONSTANT")
    max_pool = nn.MaxPool2d(kernel_size=k, stride=stride, pad_mode="valid")
    return nn.SequentialCell([pad2d, max_pool])


def gap2d(_w_in):
    """Helper for building a gap2d layer."""
    return AdaptiveAvgPool2d()


def linear(w_in, w_out, *, bias=False):
    """Helper for building a linear layer."""
    return nn.Dense(w_in, w_out, has_bias=bias, weight_init=init.Normal(sigma=0.01, mean=0.0),
                    bias_init='zeros')


def activation():
    """Helper for building an activation layer."""
    return nn.ReLU()


# ---------------------------------- Shared blocks ----------------------------------- #


class SE(nn.Cell):
    """Squeeze-and-Excitation (SE) block: AvgPool, FC, Act, FC, Sigmoid."""

    def __init__(self, w_in, w_se):
        super(SE, self).__init__()
        self.avg_pool = gap2d(w_in)
        self.f_ex = nn.SequentialCell(
            conv2d(w_in, w_se, 1, bias=True),
            activation(),
            conv2d(w_se, w_in, 1, bias=True),
            nn.Sigmoid(),
        )

    def construct(self, x):
        return x * self.f_ex(self.avg_pool(x))


# ---------------------------------- Miscellaneous ----------------------------------- #


def adjust_block_compatibility(ws, bs, gs):
    """Adjusts the compatibility of widths, bottlenecks, and groups."""
    assert len(ws) == len(bs) == len(gs)
    assert all(w > 0 and b > 0 and g > 0 for w, b, g in zip(ws, bs, gs))
    assert all(b < 1 or b % 1 == 0 for b in bs)
    vs = [int(max(1, w * b)) for w, b in zip(ws, bs)]
    gs = [int(min(g, v)) for g, v in zip(gs, vs)]
    ms = [np.lcm(g, int(b)) if b > 1 else g for g, b in zip(gs, bs)]
    vs = [max(m, int(round(v / m) * m)) for v, m in zip(vs, ms)]
    ws = [int(v / b) for v, b in zip(vs, bs)]
    assert all(w * b % g == 0 for w, b, g in zip(ws, bs, gs))
    return ws, bs, gs


class AdaptiveAvgPool2d(nn.Cell):

    def __init__(self):
        super().__init__()
        self.ReduceMean = ops.ReduceMean(keep_dims=True)

    def construct(self, x):
        return self.ReduceMean(x, (-1, -2))


def generate_regnet(w_a, w_0, w_m, d, q=8):
    """Generates per stage widths and depths from RegNet parameters."""
    assert w_a >= 0 and w_0 > 0 and w_m > 1 and w_0 % q == 0
    # Generate continuous per-block ws
    ws_cont = np.arange(d) * w_a + w_0
    # Generate quantized per-block ws
    ks = np.round(np.log(ws_cont / w_0) / np.log(w_m))
    ws_all = w_0 * np.power(w_m, ks)
    ws_all = np.round(np.divide(ws_all, q)).astype(int) * q
    # Generate per stage ws and ds (assumes ws_all are sorted)
    ws, ds = np.unique(ws_all, return_counts=True)
    # Compute number of actual stages and total possible stages
    num_stages, total_stages = len(ws), ks.max() + 1
    # Convert numpy arrays to lists and return
    ws, ds, ws_all, ws_cont = (x.tolist() for x in (ws, ds, ws_all, ws_cont))
    return ws, ds, num_stages, total_stages, ws_all, ws_cont


def generate_regnet_full(w_a, w_0, w_m, d, stride, bot_mul, group_w):
    """Generates per stage ws, ds, gs, bs, and ss from RegNet cfg."""
    ws, ds = generate_regnet(w_a, w_0, w_m, d)[0:2]
    ss = [stride for _ in ws]
    bs = [bot_mul for _ in ws]
    gs = [group_w for _ in ws]
    ws, bs, gs = adjust_block_compatibility(ws, bs, gs)
    return ws, ds, ss, bs, gs


class ResStemCifar(Cell):
    """ResNet stem for CIFAR: 3x3, BN, AF."""

    def __init__(self, w_in, w_out, bn_eps, bn_mom):
        super(ResStemCifar, self).__init__()
        self.conv = conv2d(w_in, w_out, 3)
        self.bn = norm2d(w_out, bn_eps, bn_mom)
        self.af = activation()

    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.af(x)
        # for layer in self.cells():
        #     x = layer(x)
        return x


class ResStem(Cell):
    """ResNet stem for ImageNet: 7x7, BN, AF, MaxPool."""

    def __init__(self, w_in, w_out, bn_eps, bn_mom):
        super(ResStem, self).__init__()
        self.conv = conv2d(w_in, w_out, 7, stride=2)
        self.bn = norm2d(w_out, bn_eps, bn_mom)
        self.af = activation()
        self.pool = pool2d(w_out, 3, stride=2)

    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.af(x)
        x = self.pool(x)
        # for layer in self.cells():
        #     x = layer(x)
        return x


class SimpleStem(Cell):
    """Simple stem for ImageNet: 3x3, BN, AF."""

    def __init__(self, w_in, w_out, bn_eps, bn_mom):
        super(SimpleStem, self).__init__()
        self.conv = conv2d(w_in, w_out, 3, stride=2)
        self.bn = norm2d(w_out, bn_eps, bn_mom)
        self.af = activation()

    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.af(x)
        return x


class VanillaBlock(Cell):
    """Vanilla block: [3x3 conv, BN, Relu] x2."""

    def __init__(self, w_in, w_out, stride, _params, bn_eps, bn_mom):
        super(VanillaBlock, self).__init__()
        self.a = conv2d(w_in, w_out, 3, stride=stride)
        self.a_bn = norm2d(w_out, bn_eps, bn_mom)
        self.a_af = activation()
        self.b = conv2d(w_out, w_out, 3)
        self.b_bn = norm2d(w_out, bn_eps, bn_mom)
        self.b_af = activation()

    def construct(self, x):
        x = self.a(x)
        x = self.a_bn(x)
        x = self.a_af(x)
        x = self.b(x)
        x = self.b_bn(x)
        x = self.b_af(x)
        return x


class BasicTransform(Cell):
    """Basic transformation: [3x3 conv, BN, Relu] x2."""

    def __init__(self, w_in, w_out, stride, _params, bn_eps, bn_mom):
        super(BasicTransform, self).__init__()
        self.a = conv2d(w_in, w_out, 3, stride=stride)
        self.a_bn = norm2d(w_out, bn_eps, bn_mom)
        self.a_af = activation()
        self.b = conv2d(w_out, w_out, 3)
        self.b_bn = norm2d(w_out, bn_eps, bn_mom)
        self.b_bn.final_bn = True

    def construct(self, x):
        x = self.a(x)
        x = self.a_bn(x)
        x = self.a_af(x)
        x = self.b(x)
        x = self.b_bn(x)
        return x


class ResBasicBlock(Cell):
    """Residual basic block: x + f(x), f = basic transform."""

    def __init__(self, w_in, w_out, stride, params, bn_eps, bn_mom):
        super(ResBasicBlock, self).__init__()
        self.proj, self.bn = None, None
        if (w_in != w_out) or (stride != 1):
            self.proj = conv2d(w_in, w_out, 1, stride=stride)
            self.bn = norm2d(w_out, bn_eps, bn_mom)
        self.f = BasicTransform(w_in, w_out, stride, params, bn_eps, bn_mom)
        self.af = activation()

    def construct(self, x):
        x_p = self.bn(self.proj(x)) if self.proj is not None else x
        return self.af(x_p + self.f(x))


class BottleneckTransform(Cell):
    """Bottleneck transformation: 1x1, 3x3 [+SE], 1x1."""

    def __init__(self, w_in, w_out, stride, params, bn_eps, bn_mom):
        super(BottleneckTransform, self).__init__()
        w_b = int(round(w_out * params["bot_mul"]))
        w_se = int(round(w_in * params["se_r"]))
        groups = w_b // params["group_w"]
        self.a = conv2d(w_in, w_b, 1)
        self.a_bn = norm2d(w_b, bn_eps, bn_mom)
        self.a_af = activation()
        self.b = conv2d(w_b, w_b, 3, stride=stride, groups=groups)
        self.b_bn = norm2d(w_b, bn_eps, bn_mom)
        self.b_af = activation()
        self.se = SE(w_b, w_se) if w_se else None
        self.c = conv2d(w_b, w_out, 1)
        self.c_bn = norm2d(w_out, bn_eps, bn_mom)
        self.c_bn.final_bn = True

    def construct(self, x):
        x = self.a(x)
        x = self.a_bn(x)
        x = self.a_af(x)
        x = self.b(x)
        x = self.b_bn(x)
        x = self.b_af(x)
        x = self.se(x) if self.se is not None else x
        x = self.c(x)
        x = self.c_bn(x)
        return x


class ResBottleneckBlock(Cell):
    """Residual bottleneck block: x + f(x), f = bottleneck transform."""

    def __init__(self, w_in, w_out, stride, params, bn_eps, bn_mom):
        super(ResBottleneckBlock, self).__init__()
        self.proj, self.bn = None, None
        if (w_in != w_out) or (stride != 1):
            self.proj = conv2d(w_in, w_out, 1, stride=stride)
            self.bn = norm2d(w_out, bn_eps, bn_mom)
        self.f = BottleneckTransform(w_in, w_out, stride, params, bn_eps, bn_mom)
        self.af = activation()

    def construct(self, x):
        x_p = self.bn(self.proj(x)) if self.proj is not None else x
        return self.af(x_p + self.f(x))


class ResBottleneckLinearBlock(Cell):
    """Residual linear bottleneck block: x + f(x), f = bottleneck transform."""

    def __init__(self, w_in, w_out, stride, params, bn_eps, bn_mom):
        super(ResBottleneckLinearBlock, self).__init__()
        self.has_skip = (w_in == w_out) and (stride == 1)
        self.f = BottleneckTransform(w_in, w_out, stride, params, bn_eps, bn_mom)

    def construct(self, x):
        return x + self.f(x) if self.has_skip else self.f(x)


def get_stem_fun(stem_type):
    """Retrieves the stem function by name."""
    stem_funs = {
        "res_stem_cifar": ResStemCifar,
        "res_stem_in": ResStem,
        "simple_stem_in": SimpleStem,
    }
    err_str = "Stem type '{}' not supported"
    assert stem_type in stem_funs.keys(), err_str.format(stem_type)
    return stem_funs[stem_type]


def get_block_fun(block_type):
    """Retrieves the block function by name."""
    block_funs = {
        "vanilla_block": VanillaBlock,
        "res_basic_block": ResBasicBlock,
        "res_bottleneck_block": ResBottleneckBlock,
        "res_bottleneck_linear_block": ResBottleneckLinearBlock,
    }
    err_str = "Block type '{}' not supported"
    assert block_type in block_funs.keys(), err_str.format(block_type)
    return block_funs[block_type]


class AnyStage(Cell):
    """AnyNet stage (sequence of blocks w/ the same output shape)."""

    def __init__(self, w_in, w_out, stride, d, block_fun, params, bn_eps, bn_mom):
        super(AnyStage, self).__init__()
        self.blocks = nn.CellList()
        for _ in range(d):
            block = block_fun(w_in, w_out, stride, params, bn_eps, bn_mom)
            self.blocks.append(block)
            stride, w_in = 1, w_out

    def construct(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class AnyHead(Cell):
    """AnyNet head: optional conv, AvgPool, 1x1."""

    def __init__(self, w_in, head_width, num_classes, bn_eps, bn_mom):
        super(AnyHead, self).__init__()
        self.head_width = head_width
        if head_width > 0:
            self.conv = conv2d(w_in, head_width, 1)
            self.bn = norm2d(head_width, bn_eps, bn_mom)
            self.af = activation()
            w_in = head_width
        self.avg_pool = gap2d(w_in)
        self.fc = linear(w_in, num_classes, bias=True)

    def construct(self, x):
        x = self.af(self.bn(self.conv(x))) if self.head_width > 0 else x
        x = self.avg_pool(x)
        x = x.view((x.shape[0], -1))
        x = self.fc(x)
        return x


class AnyNet(nn.Cell):
    """AnyNet model."""

    @staticmethod
    def anynet_get_params(depths, stem_type, stem_w, block_type, widths, strides, bot_muls, group_ws, head_w,
                          num_classes, se_r):
        nones = [None for _ in depths]
        return {
            "stem_type": stem_type,
            "stem_w": stem_w,
            "block_type": block_type,
            "depths": depths,
            "widths": widths,
            "strides": strides,
            "bot_muls": bot_muls if bot_muls else nones,
            "group_ws": group_ws if group_ws else nones,
            "head_w": head_w,
            "se_r": se_r,
            "num_classes": num_classes,
        }

    def __init__(self, depths, stem_type, stem_w, block_type, widths, strides, bot_muls, group_ws, head_w, num_classes,
                 bn_eps, bn_mom, se_r, in_channels):
        super(AnyNet, self).__init__()
        p = AnyNet.anynet_get_params(depths, stem_type, stem_w, block_type, widths, strides, bot_muls, group_ws, head_w,
                                     num_classes, se_r)
        stem_fun = get_stem_fun(p["stem_type"])
        block_fun = get_block_fun(p["block_type"])
        self.stem = stem_fun(in_channels, p["stem_w"], bn_eps, bn_mom)
        prev_w = p["stem_w"]
        keys = ["depths", "widths", "strides", "bot_muls", "group_ws"]
        self.stages = nn.CellList()
        for i, (d, w, s, b, g) in enumerate(zip(*[p[k] for k in keys])):
            params = {"bot_mul": b, "group_w": g, "se_r": p["se_r"]}
            stage = AnyStage(prev_w, w, s, d, block_fun, params, bn_eps, bn_mom)
            self.stages.append(stage)
            prev_w = w
        self.head = AnyHead(prev_w, p["head_w"], p["num_classes"], bn_eps, bn_mom)

    def construct(self, x):
        x = self.stem(x)
        for module in self.stages:
            x = module(x)
        x = self.head(x)
        return x


class RegNet(AnyNet):
    """RegNet model."""

    @staticmethod
    def regnet_get_params(w_a, w_0, w_m, d, stride, bot_mul, group_w, stem_type, stem_w, block_type, head_w,
                          num_classes, se_r):
        """Get AnyNet parameters that correspond to the RegNet."""
        ws, ds, ss, bs, gs = generate_regnet_full(w_a, w_0, w_m, d, stride, bot_mul, group_w)
        return {
            "stem_type": stem_type,
            "stem_w": stem_w,
            "block_type": block_type,
            "depths": ds,
            "widths": ws,
            "strides": ss,
            "bot_muls": bs,
            "group_ws": gs,
            "head_w": head_w,
            "se_r": se_r,
            "num_classes": num_classes,
        }

    def __init__(self, w_a, w_0, w_m, d, group_w, stride=2, bot_mul=1.0, stem_type='simple_stem_in', stem_w=32,
                 block_type='res_bottleneck_block', head_w=0, num_classes=1000,
                 bn_eps=1e-5, bn_mom=0.1, se_r=0, in_channels=3):
        params = RegNet.regnet_get_params(w_a, w_0, w_m, d, stride, bot_mul, group_w, stem_type, stem_w, block_type,
                                          head_w, num_classes, se_r)
        super(RegNet, self).__init__(params['depths'], params['stem_type'], params['stem_w'], params['block_type'],
                                     params['widths'], params['strides'], params['bot_muls'], params['group_ws'],
                                     params['head_w'], params['num_classes'], bn_eps, bn_mom, params['se_r'],
                                     in_channels)

    def construct(self, x):
        x = self.stem(x)
        for module in self.stages:
            x = module(x)
        x = self.head(x)
        return x


@register_model
def regnetx200mf(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs['regnetx200mf']
    model = RegNet(36.44, 24, 2.49, 13, 8, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def regnetx400mf(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs['regnetx400mf']
    model = RegNet(24.48, 24, 2.54, 22, 16, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def regnetx600mf(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs['regnetx600mf']
    model = RegNet(36.97, 48, 2.24, 16, 24, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def regnetx800mf(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs['regnetx800mf']
    model = RegNet(35.73, 56, 2.28, 16, 16, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def regnetx1_6gf(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs['regnetx1_6gf']
    model = RegNet(34.01, 80, 2.25, 18, 24, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def regnetx3_2gf(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs['regnetx3_2gf']
    model = RegNet(26.31, 88, 2.25, 25, 48, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def regnetx4_0gf(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs['regnetx4_0gf']
    model = RegNet(38.65, 96, 2.43, 23, 40, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def regnetx6_4gf(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs['regnetx6_4gf']
    model = RegNet(60.83, 184, 2.07, 17, 56, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def regnetx8_0gf(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs['regnetx8_0gf']
    model = RegNet(49.56, 80, 2.88, 23, 120, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def regnetx12gf(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs['regnetx12gf']
    model = RegNet(73.36, 168, 2.37, 19, 112, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def regnetx16gf(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs['regnetx16gf']
    model = RegNet(55.59, 216, 2.1, 22, 128, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def regnetx32gf(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs['regnetx32gf']
    model = RegNet(69.86, 320, 2.0, 23, 168, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def regnety200mf(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs['regnety200mf']
    model = RegNet(36.44, 24, 2.49, 13, 8, se_r=0.25, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def regnety400mf(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs['regnety400mf']
    model = RegNet(27.89, 48, 2.09, 16, 8, se_r=0.25, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def regnety600mf(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs['regnety600mf']
    model = RegNet(32.54, 48, 2.32, 15, 16, se_r=0.25, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def regnety800mf(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs['regnety800mf']
    model = RegNet(38.84, 56, 2.4, 14, 16, se_r=0.25, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def regnety1_6gf(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs['regnety1_6gf']
    model = RegNet(20.71, 48, 2.65, 27, 24, se_r=0.25, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def regnety3_2gf(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs['regnety3_2gf']
    model = RegNet(42.63, 80, 2.66, 21, 24, se_r=0.25, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def regnety4_0gf(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs['regnety4_0gf']
    model = RegNet(31.41, 96, 2.24, 22, 64, se_r=0.25, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def regnety6_4gf(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs['regnety6_4gf']
    model = RegNet(33.22, 112, 2.27, 25, 72, se_r=0.25, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def regnety8_0gf(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs['regnety8_0gf']
    model = RegNet(76.82, 192, 2.19, 17, 56, se_r=0.25, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def regnety12gf(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs['regnety12gf']
    model = RegNet(73.36, 168, 2.37, 19, 112, se_r=0.25, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def regnety16gf(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs['regnety16gf']
    model = RegNet(106.23, 200, 2.48, 18, 112, se_r=0.25, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def regnety32gf(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs['regnety32gf']
    model = RegNet(115.89, 232, 2.53, 20, 232, se_r=0.25, num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model
