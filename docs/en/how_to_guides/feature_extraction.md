# Multi-Scale Feature Extraction

In this guide, you will learn how to apply multi-scale feature extraction to the models in MindCV.
In real deep learning model projects, we often exploit classic CV backbones, such as ResNet, VGG, for the purposes of better performance and fast development. Generally, using only the final output of backbones is not enough.
We need outputs from intermediate layers, which act as multi-scale abstractions of the input, to help further boost the performance of our downstream tasks.
To this end, we have designed a mechanism for extracting multi-scale features from backbones in MindCV. At the time of composing this guide, MindCV has supported extracting features with this mechanism from ResNet, MobileNetV3, ConvNeXt, ResNeST, EfficientNet, RepVGG, HRNet, and ReXNet. For more details of the feature extraction mechanism, please refer to [`FeatureExtractWrapper`](https://github.com/mindspore-lab/mindcv/blob/main/mindcv/models/features.py#L36).

This guide will help you learn how to add pieces of code to extracting multi-scale features from the rest of backbones. There are mainly two steps to achieve this:

1. In `__init__()` of a model, register the intermediate layers whose outputted feature needs to be extracted in `self.feature_info`.
2. Add a wrapper function for model creation.
3. Pass `feature_only=True` and `out_indices` to [`create_model()`](https://github.com/mindspore-lab/mindcv/blob/main/mindcv/models/model_factory.py#L7).

## Layer Registration

There are mainly three possible scenarios when implementing code for feature extraction in MindCV, i.e.,
* a model with separate sequential module for each layer,
* a model with one sequential module for all layers, and
* a model with nonsequential modules.

### Scenario 1: Separate Sequential Module for Each Layer

An example of scenario 1 is shown as follows.

```
class DummyNet1(nn.Cell):
    def __init__(self, **kwargs):
        super().__init__()

        # separate sequential module for each layer
        self.layer1 = Layer()
        self.layer2 = Layer()
        self.layer3 = Layer()
        self.layer4 = Layer()

    def forward_features(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def construct(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x
```

To implement feature extraction for this scenario, we add a member variable `self.feature_info` into `__init__()` to register the extractable layers, e.g.,

```
class DummyNet1(nn.Cell):
    def __init__(self, **kwargs):
        super().__init__()
        self.feature_info = []  # for layer registration

        self.layer1 = Layer()
        self.feature_info.append(dict(chs=, reduction=, name=”layer1”))  # register layer
        self.layer2 = Layer()
        self.feature_info.append(dict(chs=, reduction=, name=”layer2”))  # register layer
        self.layer3 = Layer()
        self.feature_info.append(dict(chs=, reduction=, name=”layer3”))  # register layer
        self.layer4 = Layer()
        self.feature_info.append(dict(chs=, reduction=, name=”layer4”))  # register layer
```

As we can see above, `self.feature_info` is a list of dictionaries, each of which contains three key-value pairs. Specifically, `chs` denotes the channel number of the produced feature, `reduction` denotes the total stride at the current layer, and `name` indicates the name of this layer stored in the model parameters which can be found using [`get_parameters()`](https://www.mindspore.cn/docs/en/r1.8/note/api_mapping/pytorch_diff/GetParams.html).

For a real example of this scenario, please refer to [ResNet](https://github.com/mindspore-lab/mindcv/blob/main/mindcv/models/resnet.py#L177).

### Scenario 2: One Sequential Module for All Layers

For some models, the layers are in one sequential module. An example of scenario 2 is shown as follows.

```
class DummyNet2(nn.Cell):
    def __init__(self, **kwargs):
        super().__init__()
        layers = []

        for i in range(4):
            layers.append(Layer())

        # the layers are in one sequential module
        self.layers = nn.SequentialCell(layers)

    def forward_features(self, x):
        x = self.layers(x)
        return x

    def construct(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x
```

To implement feature extraction for this scenario, we also need to add `self.feature_info` into `__init__()` as in scenario 1, as well as create a member variable `self.flatten_sequential = True` to indicate that the sequential module in this model needs to be flattened before extracting features, e.g.,

```
class DummyNet2(nn.Cell):
    def __init__(self, **kwargs):
        super().__init__()
        self.feature_info = []  # for layer registration
        self.flatten_sequential = True  # indication of flattening the sequential module

        layers = []

        for i in range(4):
            layers.append(Layer())
			self.feature_info.append(dict(chs=, reduction=, name=f”layer{i}”))  # register layer

        self.layers = nn.SequentialCell(layers)
```

Please be reminded that the order of the module instantiations in `__init__()` is very important. The order must be kept as same as the order that these modules are called in `forward_features()` and `construct()`. Furthermore, only the modules called in `forward_features()` and `construct()` should be instantiated as member variables with the type of `nn.Cell`. Otherwise, the feature extraction mechanism will not work.

For a real example of this scenario, please refer to [MobileNetV3](https://github.com/mindspore-lab/mindcv/blob/main/mindcv/models/mobilenetv3.py#L112).

### Scenario 3: Nonsequential Modules

Layers in models sometimes are nonsequential modules. An example of scenario 3 is shown as follows.

```
class DummyNet3(nn.Cell):
    def __init__(self, **kwargs):
        super().__init__()
        layer1 = []

        for i in range(3):
                layer1.append(Layer())

        # layers in self.layer1 are not sequential
        self.layer1 = nn.CellList(layer1)

        self.stage1 = Stage()

        layer2 = []

        for i in range(3):
                layer2.append(Layer())

        # layers in self.layer2 are not sequential
        self.layer2 = nn.CellList(layer2)

        self.stage2 = Stage()

    def forward_features(self, x):
        x_list = []

        # layers are parallel instead of sequential
        for i in range(3):
                x_list.append(self.layer1[i](x))

        x = self.stage1(x_list)

        x_list = []

        # layers are parallel instead of sequential
        for i in range(3):
                x_list.append(self.layer2[i](x))

        x = self.stage2(x_list)

        return x

    def construct(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x
```

To implement feature extraction for this scenario, we need to first create a new model feature class by inheriting the original model class. Then, we add the `self.feature_info` and a member variable `self.is_rewritten = True` to indicate that this class is rewritten for feature extraction. Finally, we reimplement `forward_features()` and `construct()` with feature extraction logic. Here is an example.

```
class DummyFeatureNet3(DummyNet3):
    def __init__(self, **kwargs):
        super(DummyFeatureNet3, self).__init__(**kwargs)
        self.feature_info = []  # for layer registration
        self.is_rewritten = True  # indication of rewriting for feature extraction

        self.feature_info.append(dict(chs=, reduction=, name=”stage1”)  # register layer
        self.feature_info.append(dict(chs=, reduction=, name=”stage2”)  # register layer

    def forward_features(self, x):  # reimplement feature extraction logic
        out = []
        x_list = []

        for i in range(3):
                x_list.append(self.layer1[i](x))

        x = self.stage1(x_list)
        out.append(x)

        x_list = []

        for i in range(3):
                x_list.append(self.layer2[i](x))

        x = self.stage2(x_list)
        out.append(x)

        return out

    def construct(self, x):
        x = self.forward_features(x)
        return x
```

For a real example of this scenario, please refer to [HRNet](https://github.com/mindspore-lab/mindcv/blob/main/mindcv/models/hrnet.py#L688).

## Adding A Wrapper Function for Model Creation

After adding layer registration, we need to add one more simple wrapper function for model creation, so that the model instance can be passed to [`build_model_with_cfg()`](https://github.com/mindspore-lab/mindcv/blob/main/mindcv/models/helpers.py#L170) for feature extraction.

Usually, the original creation function of a model in MindCV looks like this,

```
@register_model
def dummynet18(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> DummyNet:
    default_cfg = default_cfgs["dummynet18"]
    model = DummyNet(..., num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model
```

As for the models falling into scenarios 1 & 2, in the wrapper function of model creation, simply pass the arguements to `build_model_with_cfg()`, e.g.,

```
def _create_dummynet(pretrained=False, **kwargs):
    return build_model_with_cfg(DummyNet, pretrained, **kwargs)

@register_model
def dummynet18(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> DummyNet:
    default_cfg = default_cfgs["dummynet18"]
    model_args = dict(..., num_classes=num_classes, in_channels=in_channels, **kwargs)
    return _create_dummynet(pretrained, **dict(default_cfg=default_cfg, **model_args))
```

As for the models falling into scenario 3, most part of the wrapper function is the same as the ones for scenarios 1 & 2. The difference lies in the part of deciding which model class to be instantiated. This is conditioned on `feature_only`, e.g.,

```
def _create_dummynet(pretrained=False, **kwargs):
    if not kwargs.get("features_only", False):
        return build_model_with_cfg(DummyNet3, pretrained, **kwargs)
    else:
        return build_model_with_cfg(DummyFeatureNet3, pretrained, **kwargs)

@register_model
def dummynet18(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> DummyNet:
    default_cfg = default_cfgs["dummynet18"]
    model_args = dict(..., num_classes=num_classes, in_channels=in_channels, **kwargs)
    return _create_dummynet(pretrained, **dict(default_cfg=default_cfg, **model_args))
```

For real examples, please refer to [ResNet](https://github.com/mindspore-lab/mindcv/blob/main/mindcv/models/resnet.py#L304) and [HRNet](https://github.com/mindspore-lab/mindcv/blob/main/mindcv/models/hrnet.py#L749).

## Passing Arguements to `create_model()`

After the previous two steps are done, we can simply create the backbone that outputs the desired features by passing `feature_only=True` and `out_indices` to `create_model()`, e.g.,

```
from mindcv.models import create_model


backbone = create_model(
    model_name="dummynet18",
    features_only=True,  # set features_only to be True
    out_indices=[0, 1, 2],  # specify the feature_info indices of the desired layers
)
```

In addtion, if we want to load a checkpoint into the backbone for feature extraction and this backbone falls into scenarios 2, we need to also set `auto_mapping=True`, e.g.,

```
from mindcv.models import create_model


backbone = create_model(
    model_name="dummynet18",
    checkpoint_path="/path/to/dummynet18.ckpt",
    auto_mapping=True,  # set auto_mapping to be True when loading a checkpoint for scenarios 2 models
    features_only=True,  # set features_only to be True
    out_indices=[0, 1, 2],  # specify the feature_info indices of the desired layers
)
```

Congradulations! Now you have learnt how to apply multi-scale feature extraction to the models in MindCV.
