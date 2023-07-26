from utils import GeneratDefaultBoxes, GridAnchorGenerator

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.common.initializer import TruncatedNormal, initializer
from mindspore.communication.management import get_group_size
from mindspore.context import ParallelMode
from mindspore.parallel._auto_parallel_context import auto_parallel_context


def _conv2d(in_channel, out_channel, kernel_size=3, stride=1, pad_mod="same"):
    return nn.Conv2d(
        in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=0, pad_mode=pad_mod, has_bias=True
    )


def _bn(channel):
    return nn.BatchNorm2d(
        channel, eps=1e-3, momentum=0.97, gamma_init=1, beta_init=0, moving_mean_init=0, moving_var_init=1
    )


def _last_conv2d(in_channel, out_channel, kernel_size=3, stride=1, pad_mod="same", pad=0):
    in_channels = in_channel
    out_channels = in_channel
    depthwise_conv = nn.Conv2d(
        in_channels, out_channels, kernel_size, stride, pad_mode=pad_mod, padding=pad, group=in_channels
    )
    conv = _conv2d(in_channel, out_channel, kernel_size=1)
    return nn.SequentialCell([depthwise_conv, _bn(in_channel), nn.ReLU6(), conv])


class ConvBNReLU(nn.Cell):
    """
    Convolution/Depthwise fused with Batchnorm and ReLU block definition.

    Args:
        in_planes (int): Input channel.
        out_planes (int): Output channel.
        kernel_size (int): Input kernel size.
        stride (int): Stride size for the first convolutional layer. Default: 1.
        groups (int): channel group. Convolution is 1 while Depthiwse is input channel. Default: 1.
        shared_conv(Cell): Use the weight shared conv, default: None.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> ConvBNReLU(16, 256, kernel_size=1, stride=1, groups=1)
    """

    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, shared_conv=None):
        super(ConvBNReLU, self).__init__()
        padding = 0
        in_channels = in_planes
        out_channels = out_planes
        if shared_conv is None:
            if groups == 1:
                conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad_mode="same", padding=padding)
            else:
                out_channels = in_planes
                conv = nn.Conv2d(
                    in_channels, out_channels, kernel_size, stride, pad_mode="same", padding=padding, group=in_channels
                )
            layers = [conv, _bn(out_planes), nn.ReLU6()]
        else:
            layers = [shared_conv, _bn(out_planes), nn.ReLU6()]
        self.features = nn.SequentialCell(layers)

    def construct(self, x):
        output = self.features(x)
        return output


class InvertedResidual(nn.Cell):
    """
    Residual block definition.

    Args:
        inp (int): Input channel.
        oup (int): Output channel.
        stride (int): Stride size for the first convolutional layer. Default: 1.
        expand_ratio (int): expand ration of input channel

    Returns:
        Tensor, output tensor.

    Examples:
        >>> ResidualBlock(3, 256, 1, 1)
    """

    def __init__(self, inp, oup, stride, expand_ratio, last_relu=False):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend(
            [
                # dw
                ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, kernel_size=1, stride=1, has_bias=False),
                _bn(oup),
            ]
        )
        self.conv = nn.SequentialCell(layers)
        self.cast = ops.Cast()
        self.last_relu = last_relu
        self.relu = nn.ReLU6()

    def construct(self, x):
        identity = x
        x = self.conv(x)

        if self.use_res_connect:
            x = identity + x

        if self.last_relu:
            x = self.relu(x)

        return x


class MobileNetV2Wrapper(nn.Cell):
    def __init__(self, backbone, args):
        super(MobileNetV2Wrapper, self).__init__()
        self.backbone = backbone
        feature1_output_channels = backbone.out_channels[0]
        self.feature1_expand_layer = ConvBNReLU(
            feature1_output_channels, int(round(feature1_output_channels * 6)), kernel_size=1
        )

        in_channels = args.extras_in_channels
        out_channels = args.extras_out_channels
        ratios = args.extras_ratio
        strides = args.extras_strides
        residual_list = []

        for i in range(2, len(in_channels)):
            residual = InvertedResidual(
                in_channels[i], out_channels[i], stride=strides[i], expand_ratio=ratios[i], last_relu=True
            )
            residual_list.append(residual)

        self.multi_residual = nn.CellList(residual_list)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        params = self.feature1_expand_layer.trainable_params()
        params.extend(self.multi_residual.trainable_params())

        for p in params:
            if "beta" not in p.name and "gamma" not in p.name and "bias" not in p.name:
                p.set_data(initializer(TruncatedNormal(0.02), p.data.shape, p.data.dtype))

    def construct(self, x):
        feature1, feature2 = self.backbone(x)
        layer_out = self.feature1_expand_layer(feature1)
        multi_feature = (layer_out, feature2)
        feature = feature2

        for residual in self.multi_residual:
            feature = residual(feature)
            multi_feature += (feature,)

        return multi_feature


class FPNTopDown(nn.Cell):
    """
    Fpn to extract features
    """

    def __init__(self, in_channel_list, out_channels):
        super(FPNTopDown, self).__init__()
        self.lateral_convs_list_ = []
        self.fpn_convs_ = []

        for channel in in_channel_list:
            l_conv = nn.Conv2d(
                channel, out_channels, kernel_size=1, stride=1, has_bias=True, padding=0, pad_mode="same"
            )
            fpn_conv = ConvBNReLU(out_channels, out_channels, kernel_size=3, stride=1)
            self.lateral_convs_list_.append(l_conv)
            self.fpn_convs_.append(fpn_conv)

        self.lateral_convs_list = nn.layer.CellList(self.lateral_convs_list_)
        self.fpn_convs_list = nn.layer.CellList(self.fpn_convs_)
        self.num_layers = len(in_channel_list)

    def construct(self, inputs):
        image_features = ()

        for i, feature in enumerate(inputs):
            image_features = image_features + (self.lateral_convs_list[i](feature),)

        features = (image_features[-1],)

        for i in range(len(inputs) - 1):
            top = len(inputs) - i - 1
            down = top - 1
            size = ops.shape(inputs[down])
            top_down = ops.ResizeBilinear((size[2], size[3]))(features[-1])
            top_down = top_down + image_features[down]
            features = features + (top_down,)

        extract_features = ()
        num_features = len(features)

        for i in range(num_features):
            extract_features = extract_features + (self.fpn_convs_list[i](features[num_features - i - 1]),)

        return extract_features


class BottomUp(nn.Cell):
    """
    Bottom Up feature extractor
    """

    def __init__(self, levels, channels, kernel_size, stride):
        super(BottomUp, self).__init__()
        self.levels = levels
        bottom_up_cells = [ConvBNReLU(channels, channels, kernel_size, stride) for x in range(self.levels)]
        self.blocks = nn.CellList(bottom_up_cells)

    def construct(self, features):
        for block in self.blocks:
            features = features + (block(features[-1]),)

        return features


class ResNet50FPNWrapper(nn.Cell):
    def __init__(self, backbone, args):
        super(ResNet50FPNWrapper, self).__init__()
        self.backbone = backbone
        self.fpn = FPNTopDown([512, 1024, 2048], 256)
        self.bottom_up = BottomUp(2, 256, 3, 2)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        params = self.fpn.trainable_params()
        params.extend(self.bottom_up.trainable_params())

        for p in params:
            if "beta" not in p.name and "gamma" not in p.name and "bias" not in p.name:
                p.set_data(initializer(TruncatedNormal(0.02), p.data.shape, p.data.dtype))

    def construct(self, x):
        feature1, feature2, feature3 = self.backbone(x)
        features = self.fpn((feature1, feature2, feature3))
        features = self.bottom_up(features)

        return features


class MobileNetV3Wrapper(nn.Cell):
    def __init__(self, backbone, args):
        super(MobileNetV3Wrapper, self).__init__()
        self.backbone = backbone

        feature1_output_channels = backbone.out_channels[0]

        self.feature1_expand_layer = nn.SequentialCell(
            [
                nn.Conv2d(feature1_output_channels, 672, 1, 1, pad_mode="pad", padding=0, has_bias=False),
                nn.BatchNorm2d(672),
                nn.HSwish(),
            ]
        )

        in_channels = args.extras_in_channels
        out_channels = args.extras_out_channels
        ratios = args.extras_ratio
        strides = args.extras_strides
        residual_list = []

        for i in range(2, len(in_channels)):
            residual = InvertedResidual(
                in_channels[i], out_channels[i], stride=strides[i], expand_ratio=ratios[i], last_relu=True
            )
            residual_list.append(residual)

        self.multi_residual = nn.CellList(residual_list)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        params = self.feature1_expand_layer.trainable_params()
        params.extend(self.multi_residual.trainable_params())

        for p in params:
            if "beta" not in p.name and "gamma" not in p.name and "bias" not in p.name:
                p.set_data(initializer(TruncatedNormal(0.02), p.data.shape, p.data.dtype))

    def construct(self, x):
        feature1, feature2 = self.backbone(x)
        layer_out = self.feature1_expand_layer(feature1)
        multi_feature = (layer_out, feature2)
        feature = feature2

        for residual in self.multi_residual:
            feature = residual(feature)
            multi_feature += (feature,)

        return multi_feature


backbone_wrapper = {
    "mobilenet_v2_100": MobileNetV2Wrapper,
    "resnet50": ResNet50FPNWrapper,
    "mobilenet_v3_large_100": MobileNetV3Wrapper,
}


class FlattenConcat(nn.Cell):
    """
    Concatenate predictions into a single tensor.

    Args:
        config (dict): The default config of SSD.

    Returns:
        Tensor, flatten predictions.
    """

    def __init__(self, args):
        super(FlattenConcat, self).__init__()
        self.num_ssd_boxes = args.num_ssd_boxes
        self.concat = ops.Concat(axis=1)
        self.transpose = ops.Transpose()

    def construct(self, inputs):
        output = ()
        batch_size = ops.shape(inputs[0])[0]

        for x in inputs:
            x = self.transpose(x, (0, 2, 3, 1))
            output += (ops.reshape(x, (batch_size, -1)),)

        res = self.concat(output)

        return ops.reshape(res, (batch_size, self.num_ssd_boxes, -1))


class MultiBox(nn.Cell):
    """
    Multibox conv layers. Each multibox layer contains class conf scores and localization predictions.

    Args:
        config (dict): The default config of SSD.

    Returns:
        Tensor, localization predictions.
        Tensor, class conf scores.
    """

    def __init__(self, args):
        super(MultiBox, self).__init__()
        num_classes = args.num_classes
        out_channels = args.extras_out_channels
        num_default = args.num_default

        loc_layers = []
        cls_layers = []

        for k, out_channel in enumerate(out_channels):
            loc_layers += [
                _last_conv2d(out_channel, 4 * num_default[k], kernel_size=3, stride=1, pad_mod="same", pad=0)
            ]
            cls_layers += [
                _last_conv2d(out_channel, num_classes * num_default[k], kernel_size=3, stride=1, pad_mod="same", pad=0)
            ]

        self.multi_loc_layers = nn.CellList(loc_layers)
        self.multi_cls_layers = nn.CellList(cls_layers)
        self.flatten_concat = FlattenConcat(args)

    def construct(self, inputs):
        loc_outputs = ()
        cls_outputs = ()

        for i in range(len(self.multi_loc_layers)):
            loc_outputs += (self.multi_loc_layers[i](inputs[i]),)
            cls_outputs += (self.multi_cls_layers[i](inputs[i]),)

        return self.flatten_concat(loc_outputs), self.flatten_concat(cls_outputs)


class WeightSharedMultiBox(nn.Cell):
    """
    Weight shared Multi-box conv layers. Each multi-box layer contains class conf scores and localization predictions.
    All box predictors shares the same conv weight in different features.

    Args:
        config (dict): The default config of SSD.
        loc_cls_shared_addition(bool): Whether the location predictor and classifier prediction share the
                                       same addition layer.
    Returns:
        Tensor, localization predictions.
        Tensor, class conf scores.
    """

    def __init__(self, args, loc_cls_shared_addition=False):
        super(WeightSharedMultiBox, self).__init__()
        num_classes = args.num_classes
        out_channels = args.extras_out_channels[0]
        num_default = args.num_default[0]
        num_features = len(args.feature_size)
        num_addition_layers = args.num_addition_layers
        self.loc_cls_shared_addition = loc_cls_shared_addition

        if not loc_cls_shared_addition:
            loc_convs = [_conv2d(out_channels, out_channels, 3, 1) for x in range(num_addition_layers)]
            cls_convs = [_conv2d(out_channels, out_channels, 3, 1) for x in range(num_addition_layers)]
            addition_loc_layer_list = []
            addition_cls_layer_list = []

            for _ in range(num_features):
                addition_loc_layer = [
                    ConvBNReLU(out_channels, out_channels, 3, 1, 1, loc_convs[x]) for x in range(num_addition_layers)
                ]
                addition_cls_layer = [
                    ConvBNReLU(out_channels, out_channels, 3, 1, 1, cls_convs[x]) for x in range(num_addition_layers)
                ]
                addition_loc_layer_list.append(nn.SequentialCell(addition_loc_layer))
                addition_cls_layer_list.append(nn.SequentialCell(addition_cls_layer))

            self.addition_layer_loc = nn.CellList(addition_loc_layer_list)
            self.addition_layer_cls = nn.CellList(addition_cls_layer_list)
        else:
            convs = [_conv2d(out_channels, out_channels, 3, 1) for x in range(num_addition_layers)]
            addition_layer_list = []

            for _ in range(num_features):
                addition_layers = [
                    ConvBNReLU(out_channels, out_channels, 3, 1, 1, convs[x]) for x in range(num_addition_layers)
                ]
                addition_layer_list.append(nn.SequentialCell(addition_layers))

            self.addition_layer = nn.SequentialCell(addition_layer_list)

        loc_layers = [_conv2d(out_channels, 4 * num_default, kernel_size=3, stride=1, pad_mod="same")]
        cls_layers = [_conv2d(out_channels, num_classes * num_default, kernel_size=3, stride=1, pad_mod="same")]

        self.loc_layers = nn.SequentialCell(loc_layers)
        self.cls_layers = nn.SequentialCell(cls_layers)
        self.flatten_concat = FlattenConcat(args)

    def construct(self, inputs):
        loc_outputs = ()
        cls_outputs = ()
        num_heads = len(inputs)

        for i in range(num_heads):
            if self.loc_cls_shared_addition:
                features = self.addition_layer[i](inputs[i])
                loc_outputs += (self.loc_layers(features),)
                cls_outputs += (self.cls_layers(features),)
            else:
                features = self.addition_layer_loc[i](inputs[i])
                loc_outputs += (self.loc_layers(features),)
                features = self.addition_layer_cls[i](inputs[i])
                cls_outputs += (self.cls_layers(features),)

        return self.flatten_concat(loc_outputs), self.flatten_concat(cls_outputs)


class SSD(nn.Cell):
    """
    SSD300 Network. Default backbone is resnet34.

    Args:
        backbone (Cell): Backbone Network.
        config (dict): The default config of SSD.

    Returns:
        Tensor, localization predictions.
        Tensor, class conf scores.

    Examples:backbone
         SSD300(backbone=resnet34(num_classes=None),
                config=config).
    """

    def __init__(self, backbone, args, is_training=True):
        super(SSD, self).__init__()
        self.backbone_wrapper = backbone_wrapper[args.backbone](backbone, args)

        if args.get("use_fpn", False):
            self.multi_box = WeightSharedMultiBox(args)
        else:
            self.multi_box = MultiBox(args)

        self.is_training = is_training

        if not is_training:
            self.activation = ops.Sigmoid()

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        params = self.multi_box.trainable_params()

        for p in params:
            if "beta" not in p.name and "gamma" not in p.name and "bias" not in p.name:
                p.set_data(initializer(TruncatedNormal(0.02), p.data.shape, p.data.dtype))

    def construct(self, x):
        multi_feature = self.backbone_wrapper(x)

        pred_loc, pred_label = self.multi_box(multi_feature)

        if not self.is_training:
            pred_label = self.activation(pred_label)

        pred_loc = ops.cast(pred_loc, ms.float32)
        pred_label = ops.cast(pred_label, ms.float32)

        return pred_loc, pred_label


class SigmoidFocalClassificationLoss(nn.Cell):
    """ "
    Sigmoid focal-loss for classification.

    Args:
        gamma (float): Hyper-parameter to balance the easy and hard examples. Default: 2.0
        alpha (float): Hyper-parameter to balance the positive and negative example. Default: 0.25

    Returns:
        Tensor, the focal loss.
    """

    def __init__(self, gamma=2.0, alpha=0.25):
        super(SigmoidFocalClassificationLoss, self).__init__()
        self.sigmiod_cross_entropy = ops.SigmoidCrossEntropyWithLogits()
        self.sigmoid = ops.Sigmoid()
        self.pow = ops.Pow()
        self.onehot = ops.OneHot()
        self.on_value = Tensor(1.0, ms.float32)
        self.off_value = Tensor(0.0, ms.float32)
        self.gamma = gamma
        self.alpha = alpha

    def construct(self, logits, label):
        label = self.onehot(label, ops.shape(logits)[-1], self.on_value, self.off_value)
        sigmiod_cross_entropy = self.sigmiod_cross_entropy(logits, label)
        sigmoid = self.sigmoid(logits)
        label = ops.cast(label, ms.float32)
        p_t = label * sigmoid + (1 - label) * (1 - sigmoid)
        modulating_factor = self.pow(1 - p_t, self.gamma)
        alpha_weight_factor = label * self.alpha + (1 - label) * (1 - self.alpha)
        focal_loss = modulating_factor * alpha_weight_factor * sigmiod_cross_entropy
        return focal_loss


class SSDWithLossCell(nn.Cell):
    """ "
    Provide SSD training loss through network.

    Args:
        network (Cell): The training network.
        config (dict): SSD config.

    Returns:
        Tensor, the loss of the network.
    """

    def __init__(self, network, args):
        super(SSDWithLossCell, self).__init__(auto_prefix=False)
        self.network = network
        self.less = ops.Less()
        self.tile = ops.Tile()
        self.reduce_sum = ops.ReduceSum()
        self.expand_dims = ops.ExpandDims()
        self.class_loss = SigmoidFocalClassificationLoss(args.gamma, args.alpha)
        self.loc_loss = nn.SmoothL1Loss()

    def construct(self, x, gt_loc, gt_label, num_matched_boxes):
        pred_loc, pred_label = self.network(x)
        mask = ops.cast(self.less(0, gt_label), ms.float32)
        num_matched_boxes = self.reduce_sum(ops.cast(num_matched_boxes, ms.float32))

        # Localization Loss
        mask_loc = self.tile(self.expand_dims(mask, -1), (1, 1, 4))
        smooth_l1 = self.loc_loss(pred_loc, gt_loc) * mask_loc
        loss_loc = self.reduce_sum(self.reduce_sum(smooth_l1, -1), -1)

        # Classification Loss
        loss_cls = self.class_loss(pred_label, gt_label)
        loss_cls = self.reduce_sum(loss_cls, (1, 2))

        return self.reduce_sum((loss_cls + loss_loc) / num_matched_boxes)


grad_scale = ops.MultitypeFuncGraph("grad_scale")


@grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * ops.Reciprocal()(scale)


class TrainingWrapper(nn.Cell):
    """
    Encapsulation class of SSD network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        sens (Number): The adjust parameter. Default: 1.0.
        use_global_nrom(bool): Whether apply global norm before optimizer. Default: False
    """

    def __init__(self, network, optimizer, sens=1.0, use_global_norm=False):
        super(TrainingWrapper, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.weights = ms.ParameterTuple(network.trainable_params())
        self.optimizer = optimizer
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer = None
        self.use_global_norm = use_global_norm
        self.parallel_mode = ms.get_auto_parallel_context("parallel_mode")

        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True

        if self.reducer_flag:
            mean = ms.get_auto_parallel_context("gradients_mean")

            if auto_parallel_context().get_device_num_is_set():
                degree = ms.get_auto_parallel_context("device_num")
            else:
                degree = get_group_size()

            self.grad_reducer = nn.DistributedGradReducer(optimizer.parameters, mean, degree)

        self.hyper_map = ops.HyperMap()

    def construct(self, *args):
        weights = self.weights
        loss = self.network(*args)
        sens = ops.Fill()(ops.DType()(loss), ops.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(*args, sens)

        if self.reducer_flag:
            # apply grad reducer on grads
            grads = self.grad_reducer(grads)

        if self.use_global_norm:
            grads = self.hyper_map(ops.partial(grad_scale, ops.scalar_to_tensor(self.sens)), grads)
            grads = ops.clip_by_global_norm(grads)

        self.optimizer(grads)

        return loss


class SSDInferWithDecoder(nn.Cell):
    """
    SSD Infer wrapper to decode the bbox locations.

    Args:
        network (Cell): the origin ssd infer network without bbox decoder.
        default_boxes (Tensor): the default_boxes from anchor generator
        config (dict): ssd config
    Returns:
        Tensor, the locations for bbox after decoder representing (y0,x0,y1,x1)
        Tensor, the prediction labels.

    """

    def __init__(self, network, args):
        super(SSDInferWithDecoder, self).__init__(auto_prefix=False)
        self.network = network

        if hasattr(args, "use_anchor_generator") and args.use_anchor_generator:
            self.default_boxes, _ = GridAnchorGenerator(args.image_size, 4, 2, [1.0, 2.0, 0.5]).generate_multi_levels(
                args.steps
            )
            self.default_boxes = Tensor(self.default_boxes)
        else:
            self.default_boxes = Tensor(GeneratDefaultBoxes(args).default_boxes)

        self.prior_scaling_xy = args.prior_scaling[0]
        self.prior_scaling_wh = args.prior_scaling[1]

    def construct(self, x):
        pred_loc, pred_label = self.network(x)

        default_bbox_xy = self.default_boxes[..., :2]
        default_bbox_wh = self.default_boxes[..., 2:]
        pred_xy = pred_loc[..., :2] * self.prior_scaling_xy * default_bbox_wh + default_bbox_xy
        pred_wh = ops.Exp()(pred_loc[..., 2:] * self.prior_scaling_wh) * default_bbox_wh

        pred_xy_0 = pred_xy - pred_wh / 2.0
        pred_xy_1 = pred_xy + pred_wh / 2.0
        pred_xy = ops.Concat(-1)((pred_xy_0, pred_xy_1))
        pred_xy = ops.Maximum()(pred_xy, 0)
        pred_xy = ops.Minimum()(pred_xy, 1)
        return pred_xy, pred_label


def get_ssd_trainer(model, optimizer, args):
    return ms.Model(TrainingWrapper(model, optimizer, args.loss_scale))
