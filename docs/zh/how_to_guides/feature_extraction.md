# 多尺度特征提取

在本教程中，您将学习如何对MindCV中的模型进行多尺度特征抽取。
在实际的深度学习模型项目中，我们经常使用经典的计算机视觉骨干网络，如ResNet、VGG，以获得更好的性能和更快的开发进度。通常情况下，仅使用骨干网络的最终输出往往是不够的。
我们需要来自中间层的输出，它们作为输入的多尺度抽象，可以帮助进一步提升我们下游任务的性能。
为此，我们在MindCV中设计了一种从骨干网络中抽取多尺度特征的机制。在编写本教程时，MindCV已经支持从ResNet、MobileNetV3、ConvNeXt、ResNeST、EfficientNet、RepVGG、HRNet和ReXNet中使用这种机制抽取特征。有关特征抽取机制的更多详细信息，请参阅[`FeatureExtractWrapper`](https://github.com/mindspore-lab/mindcv/blob/main/mindcv/models/features.py#L36).

本教程将帮助您学习如何添加一些代码来从其余骨干网络中抽取多尺度特征。主要有两个步骤：

1. 在模型的`__init__()`中，注册需要提取输出特征的中间层，将其添加到`self.feature_info`中。
2. 添加一个用于模型创建的封装函数。
3. 将参数`feature_only=True`和`out_indices`传入[`create_model()`](https://github.com/mindspore-lab/mindcv/blob/main/mindcv/models/model_factory.py#L7).

## 注册中间层

在MindCV中实现特征抽取代码时，主要有三种模型实现情况，即：
* 对于每个中间层，模型都有单独对应的顺序模块。
* 模型所有层都包含在同一个顺序模块中。
* 模型中间层是非顺序模块。

### 场景1：每个中间层有单独对应的顺序模块

场景1的示例如下。

```
class DummyNet1(nn.Cell):
    def __init__(self, **kwargs):
        super().__init__()

        # 每个层都有单独的顺序模块
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

为了实现场景1的特征抽取，我们在`__init__()`中添加了成员变量`self.feature_info`用于注册可提取的中间层，例如，

```
class DummyNet1(nn.Cell):
    def __init__(self, **kwargs):
        super().__init__()
        self.feature_info = []   # 用于中间层注册

        self.layer1 = Layer()
        self.feature_info.append(dict(chs=, reduction=, name=”layer1”))  # 注册中间层
        self.layer2 = Layer()
        self.feature_info.append(dict(chs=, reduction=, name=”layer2”))  # 注册中间层
        self.layer3 = Layer()
        self.feature_info.append(dict(chs=, reduction=, name=”layer3”))  # 注册中间层
        self.layer4 = Layer()
        self.feature_info.append(dict(chs=, reduction=, name=”layer4”))  # 注册中间层
```

如上所示，`self.feature_info`是一个包含多个字典的列表，每个字典包含三对键值对。具体而言, `chs` 表示生成特征的通道数，`reduction`表示当前层的总`stide`数，`name`表示该中间层在模型参数中的名称。 此名称可用[`get_parameters()`](https://www.mindspore.cn/docs/en/r1.8/note/api_mapping/pytorch_diff/GetParams.html)找到。

有关此场景的真实示例，请参阅[ResNet](https://github.com/mindspore-lab/mindcv/blob/main/mindcv/models/resnet.py#L177)。

### 场景2：所有中间层在同一个顺序模块中

对于某些模型，所有中间层都被包含在同一个顺序模块中。场景2的示例如下。

```
class DummyNet2(nn.Cell):
    def __init__(self, **kwargs):
        super().__init__()
        layers = []

        for i in range(4):
            layers.append(Layer())

        # 所有中间层在同一个顺序模块中
        self.layers = nn.SequentialCell(layers)

    def forward_features(self, x):
        x = self.layers(x)
        return x

    def construct(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x
```

为了实现场景2的特征抽取，我们同样需要在`__init__()`中添加`self.feature_info`，同时还要创建一个成员变量`self.flatten_sequential = True`，用于表示在提取特征之前需要将此模型中的顺序模块展开，例如，

```
class DummyNet2(nn.Cell):
    def __init__(self, **kwargs):
        super().__init__()
        self.feature_info = []  # 用于中间层注册
        self.flatten_sequential = True  # 表示需要展开顺序模块

        layers = []

        for i in range(4):
            layers.append(Layer())
			self.feature_info.append(dict(chs=, reduction=, name=f”layer{i}”))  # 注册中间层

        self.layers = nn.SequentialCell(layers)
```

请注意，`__init__()`中模块实例化的顺序非常重要。顺序必须与在`forward_features()`和`construct()`中调用这些模块的顺序保持一致。此外，只有在`forward_features()`和`construct()`中被调用的`nn.Cell`类型模块，才能作为成员变量被实例化。否则，特征抽取机制将无法正常工作。

有关此场景的真实示例，请参阅[MobileNetV3](https://github.com/mindspore-lab/mindcv/blob/main/mindcv/models/mobilenetv3.py#L112)。

### 场景3：模型中间层为非顺序模块

模型中间层有时是非顺序模块。场景3的示例如下。

```
class DummyNet3(nn.Cell):
    def __init__(self, **kwargs):
        super().__init__()
        layer1 = []

        for i in range(3):
                layer1.append(Layer())

        # self.layer1 中的层不是顺序模块
        self.layer1 = nn.CellList(layer1)

        self.stage1 = Stage()

        layer2 = []

        for i in range(3):
                layer2.append(Layer())

        # self.layer2 中的层不是顺序模块
        self.layer2 = nn.CellList(layer2)

        self.stage2 = Stage()

    def forward_features(self, x):
        x_list = []

        # 中间层是并行的，而不是顺序的
        for i in range(3):
                x_list.append(self.layer1[i](x))

        x = self.stage1(x_list)

        x_list = []

        # 中间层是并行的，而不是顺序的
        for i in range(3):
                x_list.append(self.layer2[i](x))

        x = self.stage2(x_list)

        return x

    def construct(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x
```

为了实现场景3的特征抽取，我们需要首先通过继承原始模型类创建一个新的模型特征类。然后，在`__init__()`中添加`self.feature_info`和一个成员变量`self.is_rewritten = True`，以指示该类是为特征抽取而重写的。最后，我们使用特征抽取逻辑重新实现`forward_features()`和`construct()`。以下是一个示例。

```
class DummyFeatureNet3(DummyNet3):
    def __init__(self, **kwargs):
        super(DummyFeatureNet3, self).__init__(**kwargs)
        self.feature_info = []  # 用于中间层注册
        self.is_rewritten = True  # 表示为特征抽取而重写的

        self.feature_info.append(dict(chs=, reduction=, name=”stage1”)  # 注册中间层
        self.feature_info.append(dict(chs=, reduction=, name=”stage2”)  # 注册中间层

    def forward_features(self, x):  # 重新实现特征抽取逻辑
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

有关此情况的真实示例，请参阅[HRNet](https://github.com/mindspore-lab/mindcv/blob/main/mindcv/models/hrnet.py#L688)。

## 添加模型创建的封装函数

在添加中间层注册后，我们需要再添加一个简单的模型创建封装函数，以便将模型实例传递给[`build_model_with_cfg()`](https://github.com/mindspore-lab/mindcv/blob/main/mindcv/models/helpers.py#L170)进行特征抽取。

通常，MindCV中模型的原始创建函数如下所示：

```
@register_model
def dummynet18(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> DummyNet:
    default_cfg = default_cfgs["dummynet18"]
    model = DummyNet(..., num_classes=num_classes, in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model
```

对于属于场景1和场景2的模型，在添加的模型创建封装函数中，只需将原来的参数传给`build_model_with_cfg()`即可，例如，

```
def _create_dummynet(pretrained=False, **kwargs):
    return build_model_with_cfg(DummyNet, pretrained, **kwargs)

@register_model
def dummynet18(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> DummyNet:
    default_cfg = default_cfgs["dummynet18"]
    model_args = dict(..., num_classes=num_classes, in_channels=in_channels, **kwargs)
    return _create_dummynet(pretrained, **dict(default_cfg=default_cfg, **model_args))
```

对于属于场景3的模型，封装函数大致与场景1和场景2类似。不同之处在于需要决定实例化哪个模型类，这取决于`feature_only`。例如，

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

有关实际示例，请参阅[ResNet](https://github.com/mindspore-lab/mindcv/blob/main/mindcv/models/resnet.py#L304)与[HRNet](https://github.com/mindspore-lab/mindcv/blob/main/mindcv/models/hrnet.py#L749)。

## `create_model()`传入参数

完成前面两个步骤后，我们可将`feature_only=True`和`out_indices`传递给`create_model()`来创建可输出所需特征的骨干网络，例如，

```
from mindcv.models import create_model


backbone = create_model(
    model_name="dummynet18",
    features_only=True,  # 设置features_only为 True
    out_indices=[0, 1, 2],  # 指定特征抽取的中间层在feature_info中的索引
)
```

此外，如果我们想要将checkpoint加载到用于特征抽取的骨干网络中，并且此骨干网络属于场景2，那么，我们还需设置`auto_mapping=True`，例如，

```
from mindcv.models import create_model


backbone = create_model(
    model_name="dummynet18",
    checkpoint_path="/path/to/dummynet18.ckpt",
    auto_mapping=True,  # 当为场景2的模型加载checkpoint时，将auto_mapping设为True
    features_only=True,  # 设置features_only为 True
    out_indices=[0, 1, 2],  # 指定特征抽取的中间层在feature_info中的索引
)
```

恭喜您！现在您已经学会了如何对MindCV中的模型进行多尺度特征抽取。
