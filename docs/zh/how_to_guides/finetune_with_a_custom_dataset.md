# 自定义数据集的模型微调指南

本文档提供了在自定义数据集上微调MindCV预训练模型的参考流程以及在线读取数据集、分层设置学习率、冻结部分特征网络等微调技巧的实现方法，主要代码实现集成在./example/finetune.py中，您可以基于此教程根据需要自行改动。

接下来将以FGVC-Aircraft数据集为例展示如何对预训练模型mobilenet v3-small进行微调。[Fine-Grained Visual Classification of Aircraft](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/)是常用的细粒度图像分类基准数据集，包含 10000 张飞机图片，100 种不同的飞机型号(variant)，其中每种飞机型号均有 100 张图片。

首先将下载后的数据集解压到./data文件夹下，Aircraft数据集的目录为：

```text
aircraft
└── data
    ├── images
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ....
    ├──	images_variant_test.txt
    ├──	images_variant_trainval.txt
    └── ....
```

其中images文件夹包含全部10000张图片，每张图片所属的飞机型号和子集由images_variant_*.txt标注。在模型微调阶段，训练集一般由images_variant_trainval.txt 确定。经过拆分后，训练集应当包含6667张图片，测试集包含3333张图片。

## 数据预处理

### 读取数据集

对于自定义数据集而言，既可以先在本地将数据文件目录整理成与ImageNet类似的树状结构，再使用`create_dataset`读取数据集（离线方式，仅适用于小型数据集），又可以直接[将原始数据集读取成可迭代/可映射对象]((https://www.mindspore.cn/tutorials/en/r2.1/beginner/dataset.html#customizing-dataset))，替代文件拆分与`create_dataset`步骤（在线方式）。

#### 离线方式

MindCV的`create_dataset`接口使用[`mindspore.dataset.ImageFolderDataset`](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/dataset/mindspore.dataset.ImageFolderDataset.html#mindspore.dataset.ImageFolderDataset)函数构建数据对象，同一个文件夹内的所有图片将会根据文件夹名字被分配相同的标签。因此，使用该流程的前提条件是源数据集的文件目录应当遵循如下树状结构：

```text
DATASET_NAME
    ├── split1(e.g. train)/
    │  ├── class1/
    │  │   ├── 000001.jpg
    │  │   ├── 000002.jpg
    │  │   └── ....
    │  └── class2/
    │      ├── 000001.jpg
    │      ├── 000002.jpg
    │      └── ....
    └── split2/
       ├── class1/
       │   ├── 000001.jpg
       │   ├── 000002.jpg
       │   └── ....
       └── class2/
           ├── 000001.jpg
           ├── 000002.jpg
           └── ....
```

接下来以说明文件./aircraft/data/images_variant_trainval.txt 为例，在本地生成满足前述树状结构的训练集文件 ./aircraft/data/images/trainval/。

```python
""" Extract images and generate ImageNet-style dataset directory """
import os
import shutil


# only for Aircraft dataset but not a general one
def extract_images(images_path, subset_name, annotation_file_path, copy=True):
    # read the annotation file to get the label of each image
    def annotations(annotation_file_path):
        image_label = {}
        with open(annotation_file_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                label = " ".join(line.split(" ")[1:]).replace("\n", "").replace("/", "_")
                if label not in image_label.keys():
                    image_label[label] = []
                    image_label[label].append(line.split(" ")[0])
                else:
                    image_label[label].append(line.split(" ")[0])
        return image_label

    # make a new folder for subset
    subset_path = images_path + subset_name
    os.mkdir(subset_path)

    # extract and copy/move images to the new folder
    image_label = annotations(annotation_file_path)
    for label in image_label.keys():
        label_folder = subset_path + "/" + label
        os.mkdir(label_folder)
        for image in image_label[label]:
            image_name = image + ".jpg"
            if copy:
                shutil.copy(images_path + image_name, label_folder + image_name)
            else:
                shutil.move(images_path + image_name, label_folder)


# take train set of aircraft dataset as an example
images_path = "./aircraft/data/images/"
subset_name = "trainval"
annotation_file_path = "./aircraft/data/images_variant_trainval.txt"
extract_images(images_path, subset_name, annotation_file_path)
```

测试集的拆分方式与训练集一致，整理完成的Aircraft数据集文件结构应为：

```text
aircraft
└── data
	└── images
        ├── trainval
        │   ├── 707-320
        │   │   ├── 0056978.jpg
        │   │   └── ....
        │   ├── 727-200
        │   │   ├── 0048341.jpg
        │   │	└── ....
        │  	└── ....
        └── test
            ├── 707-320
            │   ├── 0062765.jpg
            │   └── ....
            ├── 727-200
            │   ├── 0061581.jpg
            │   └── ....
            └── ....
```

由于模型微调文件./example/finetune.py中集成了`create_dataset`->`create_transforms`->`create_loader`->`create_model`->...等所有从预处理到建立、验证模型的训练流程，使用离线方式整理完文件目录结构的数据集可以**直接通过运行`python ./example/finetune.py`命令完成后续读取数据与训练模型**这一整套操作。对于自定义数据集而言，还需<font color=DarkRed>注意提前将配置文件中的`dataset`参数设置为空字符串`""`</font>。

#### 在线方式

离线方式的数据读取会在本地占用额外的磁盘空间存储新生成的数据文件，因此在本地存储空间不足或无法将数据备份到本地等其他特殊情况下，无法直接使用`create_dataset`接口读取本地数据文件时，可以采用在线方式自行编写函数读取数据集。

以生成储存训练集图片和索引到图片样本映射的可随机访问数据集为例：

- 首先定义一个读取原始数据并将其转换成可随机访问的数据集对象`ImageClsDataset`：

	- 在该类的初始化函数`__init__()`中，以./aircraft/data/images_variant_trainval.txt为例的标注文件的文件路径将被当做输入，用于生成储存图片与标签一一对应关系的字典`self.annotation`；

	- 由于在`create_loader`中将会对此对象进行map操作，而该操作不支持字符串格式的标签，因此还需要生成`self.label2id`并<font color=DarkRed>将`self.annotation`中字符串格式的标签转换成整数格式</font>；

	- 根据`self.annotation`中储存的信息，从文件夹./aircraft/data/images/中<font color=DarkRed>将训练集图片读取成一维数组形式</font>（由于`create_loader`中map操作限制，此处图片数据必须被读取为一维格式），并将图片信息与标签分别存放到`self._data`与`self._label`中；

	- 接下来使用`__getitem__`方法构造可随机访问的数据集对象。

-
  构造完`ImageClsDataset`类之后，向其传入标注文件的路径以实例化该类，并通过[`mindspore.dataset.GeneratorDataset`](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/dataset/mindspore.dataset.GeneratorDataset.html#mindspore.dataset.GeneratorDataset)函数将该可映射对象加载成数据集即可，注意该函数的参数`column_names`必须被设置为["image", "label"]以便后续其他接口读取数据，此时得到的`dataset_train`应当与通过`create_dataset`读取的训练集完全一致。


```python
import numpy as np

from mindspore.dataset import GeneratorDataset


class ImageClsDataset:
    def __init__(self, annotation_dir, images_dir):
        # Read annotations
        self.annotation = {}
        with open(annotation_dir, "r") as f:
            lines = f.readlines()
            for line in lines:
                image_label = line.replace("\n", "").replace("/", "_").split(" ")
                image = image_label[0] + ".jpg"
                label = " ".join(image_label[1:])
                self.annotation[image] = label

        # Transfer string-type label to int-type label
        self.label2id = {}
        labels = sorted(list(set(self.annotation.values())))
        for i in labels:
            self.label2id[i] = labels.index(i)

        for image, label in self.annotation.items():
            self.annotation[image] = self.label2id[label]

        # Read image-labels as mappable object
        label2images = {key: [] for key in self.label2id.values()}
        for image, label in self.annotation.items():
            read_image = np.fromfile(images_dir + image, dtype=np.uint8)
            label2images[label].append(read_image)

        self._data = sum(list(label2images.values()), [])
        self._label = sum([[i] * len(label2images[i]) for i in label2images.keys()], [])

    # make class ImageClsDataset a mappable object
    def __getitem__(self, index):
        return self._data[index], self._label[index]

    def __len__(self):
        return len(self._data)


# take aircraft dataset as an example
annotation_dir = "./aircraft/data/images_variant_trainval.txt"
images_dir = "./aircraft/data/images/"
dataset = ImageClsDataset(annotation_dir, images_dir)
dataset_train = GeneratorDataset(source=dataset, column_names=["image", "label"], shuffle=True)

```

与离线方式读取数据集相比，在线读取方式省略了在本地拆分数据文件并用`create_dataset`接口读取本地文件的步骤，因此在后续的训练中，只需**将finetune.py中使用`create_dataset`接口的部分替换成上述代码**，就可以与离线方式一样，直接运行finetune.py开始训练。

### 数据增强与分批

MindCV使用`create_loader`函数对上一章节读取的数据集进行图像增强与分批处理，图像增强策略通过`create_transforms`函数事先定义，分批处理操作通过`create_loader`函数中的参数`batch_size`定义，以上涉及到的**所有超参数均可以通过模型配置文件传递**，超参数具体使用方法见[API说明](https://mindspore-lab.github.io/mindcv/zh/reference/data/)。

对于规模较小的自定义数据集，建议可以在这一部分对训练集做额外的数据增强处理，以增强模型的泛化性，防止过拟合。对于细粒度图像分类任务的数据集，比如本文中的Aircraft数据集，由于数据类内方差较大可能导致分类效果较差，还可以通过调整超参数`image_resize`适当增大图片尺寸（如：448、512、600等等）。

## 模型微调

参考[Stanford University CS231n](https://cs231n.github.io/transfer-learning/#tf)，**整体微调**、**冻结特征网络微调**、与**分层设置学习率微调**是常用的微调模式。模型的整体微调使用预训练权重初始化目标模型的参数并在此基础上针对新数据集继续训练、更新所有参数，因此计算量较大，耗时较长但一般精度较高；冻结特征网络则分为冻结所有特征网络与冻结部分特征网络两种，前者将预训练模型作为特征提取器，仅更新全连接层参数，耗时短但精度低，后者一般固定学习基础特征的浅层参数，只更新学习精细特征的深层网络参数与全连接层参数；分层设置学习率与之相似，但是更加精细地指定了网络内部某些特定层在训练中更新参数所使用的学习率。

对于实际微调训练中所使用的的超参数配置，可以参考./configs中基于ImageNet-1k数据集预训练的配置文件。注意应事先<font color=DarkRed>将`num_classes`设置为自定义数据集的标签个数</font>（比如Aircfrat数据集是100）；<font color=DarkRed>将超参数`pretrained`设置为`True`以自动下载与加载预训练权重</font>，过程中由于`num_classes`并非默认的1000，分类层的参数将会被自动去除（此时无需设置`ckpt_path`,但如果您需要加载来自本地的checkpoint文件，则需要保持`pretrained`为`False`并自行指定`ckpt_path`参数，注意务必提前自行将checkpoint文件中分类层的参数剔除）。此外，还可以基于自定义数据集规模，<font color=DarkRed>适当调小`batch_size`与`epoch_size`</font>，由于预训练权重中已经包含了许多识别图像的初始信息，为了不过分破坏这些信息，还需<font color=DarkRed>将学习率`lr`调小</font>，建议至多从预训练学习率的十分之一或0.0001开始训练、调参。这些参数都可以在配置文件中修改，也可以如下所示在shell命令中添加，训练结果可在./ckpt/results.txt文件中查看。

```bash
python .examples/finetune/finetune.py --config=./configs/mobilenetv3/mobilnet_v3_small_ascend.yaml --data_dir=./aircraft/data --num_classes=100 --pretrained=True ...
```

本文在基于Aircraft数据集对mobilenet v3-small微调时主要对超参数做了如下改动：

| Hyper-parameter | Pretrain   | Fine-tune            |
| --------------- |------------|----------------------|
| dataset         | "imagenet" | ""                   |
| batch_size      | 75         | 8                    |
| image_resize    | 224        | 600                  |
| auto_augment    | -          | "randaug-m7-mstd0.5" |
| num_classes     | 1000       | 100                  |
| pretrained      | False      | True                 |
| epoch_size      | 470        | 50                   |
| lr              | 0.77       | 0.002                |

### 整体微调

由于整体微调的训练流程与从头训练一致，因此只需通过**运行finetune.py启动训练**并跟从头训练一样调参即可。

### 冻结特征网络

#### 冻结所有特征网络

我们通过对除全连接层外的所有参数设置`requires_grad=False`来防止其参数更新。在finetune.py中，只需在创建模型`create_model`之后加入如下代码即可实现：

```python
from mindcv.models.registry import _model_pretrained_cfgs

# ...create_model()

# number of parameters to be updated
num_params = 2

# read names of parameters in FC layer
classifier_names = [_model_pretrained_cfgs[args.model]["classifier"] + ".weight",
                    _model_pretrained_cfgs[args.model]["classifier"] + ".bias"]

# prevent parameters in network(except the classifier) from updating
for param in network.trainable_params():
    if param.name not in classifier_names:
        param.requires_grad = False
```

#### 冻结部分特征网络

为了平衡微调训练的速度和精度，我们还可以固定部分目标网络参数，有针对性地训练网络中的深层参数。实现这一操作只需要提取出要冻结的层中的参数名称，并在上述冻结所有特征网络的代码基础上稍作修改即可。通过打印`create_model`的结果——`network`可知，MindCV中对mobilenet v3-small的每层网络命名为`"features.*"`，假设我们仅冻结网络前7层，在finetune.py中创建模型`create_model`后加入如下代码即可：

```python
# ...create_model()

# read names of network layers
freeze_layer=["features."+str(i) for i in range(7)]

# prevent parameters in first 7 layers of network from updating
for param in network.trainable_params():
    for layer in freeze_layer:
        if layer in param.name:
            param.requires_grad = False
```

### 分层设置学习率

为了进一步提升微调网络的训练效果，还可以分层设置训练中的学习率。这是由于浅层网络一般是识别通用的轮廓特征，所以即便重新更新该部分参数，学习率也应该被设置得比较小；深层部分一般识别物体精细的个性特征，学习率也因此可以设置得比较大；而相对于需要尽量保留预训练信息的特征网络而言，分类器需要从头开始训练，也可以适当将学习率调大。由于针对特定网络层的学习率调整操作比较精细，我们需要进入finetune.py中自行指定参数名与对应的学习率。

MindCV使用[`create_optimizer`](https://mindspore-lab.github.io/mindcv/zh/reference/optim/#mindcv.optim.optim_factory.create_optimizer)函数构造优化器，并将学习率传到优化器中去。要设置分层学习率，只需**将finetune.py中`create_optimizer`函数的`params`参数从`network.trainable_params()`改为包含特定层参数名与对应学习率的列表即可**，参考[MindSpore各优化器说明文档](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/mindspore.nn.html#%E4%BC%98%E5%8C%96%E5%99%A8)，其中网络具体结构与每层中的参数名均可以通过打印`create_model`的结果——`network`查看。
> Tips: 您还可以使用同样的操作分层设置weight_decay.

#### 单独调整分类器的学习率

以mobilenet v3-small为例，该模型分类器名称以“classifier”开头，因此如果仅调大分类器的学习率，我们需要指定分类器在每一步训练中的学习率。`lr_scheduler`是由`create_scheduler`生成的学习率调整策略，是一个包含网络每步训练中具体学习率值的列表，假设我们将分类器的学习率调整至特征网络学习率的1.2倍，finetune.py中创建优化器部分代码的改动如下：

```python
# ...


# Note: a)the params-lr dict must contain all the parameters. b)Also, you're recommended to set a dict with a key "order_params" to make sure the parameters will be updated in a right order.
params_lr_group = [{"params": list(filter(lambda x: 'classifier' in x.name, network.trainable_params())),
                    "lr": [i*1.2 for i in lr_scheduler]},
                   {"params": list(filter(lambda x: 'classifier' not in x.name, network.trainable_params())),
                    "lr": lr_scheduler},
                   {"order_params": network.trainable_params()}]

optimizer = create_optimizer(params_lr_group,
                             opt=args.opt,
                             lr=lr_scheduler,
                             ...)
```

#### 设置特征网络任意层的学习率

与单独调整分类器的学习率类似，分层设置特征网络学习率需要指定特定层的学习率变化列表。假设我们仅增大特征网络最后三层参数（features.13, features.14, features.15）更新的学习率，对finetune.py中创建优化器部分代码的改动如下：

```python
# ...


# Note: a)the params-lr dict must contain all the parameters. b)Also, you're recommended to set a dict with a key "order_params" to make sure the parameters will be updated in a right order.
params_lr_group = [{"params": list(filter(lambda x: 'features.13' in x.name, network.trainable_params())),
                    "lr": [i * 1.05 for i in lr_scheduler]},
                   {"params": list(filter(lambda x: 'features.14' in x.name, network.trainable_params())),
                    "lr": [i * 1.1 for i in lr_scheduler]},
                   {"params": list(filter(lambda x: 'features.15' in x.name, network.trainable_params())),
                    "lr": [i * 1.15 for i in lr_scheduler]},
                   {"params": list(filter(
                       lambda x: ".".join(x.name.split(".")[:2]) not in ["features.13", "features.14", "features.15"],
                       network.trainable_params())),
                    "lr": lr_scheduler},
                   {"order_params": network.trainable_params()}]

optimizer = create_optimizer(params_lr_group,
                             opt=args.opt,
                             lr=lr_scheduler,
                             ...)
```


## 模型评估

训练结束后，使用./ckpt文件夹中以`*_best.ckpt`格式储存的模型权重来评估模型在测试集上的最优表现，只需**直接运行validate.py**并向其传入模型配置文件路径与权重的文件路径即可：

```bash
python validate.py --config=./configs/mobilenetv3/mobilnet_v3_small_ascend.yaml --data_dir=./aircraft/data --ckpt_path=./ckpt/mobilenet_v3_small_100_best.ckpt
```

模型微调章节展示了多种微调技巧，下表总结了在使用相同训练配置不同微调方式下mobilenet v3-small模型在Aircraft数据集上的Top-1 精度表现：

| 模型               | 冻结所有特征网络 | 冻结浅层特征网络 | 全量微调+固定学习率 | 全量微调+调大分类器学习率 |      | 全量微调+调大深层网络学习率 |
| ------------------ | ---------------- | ---------------- | ------------------- | ------------------------- | ---- | --------------------------- |
| mobilenet v3-small | 48.66%           | 76.83%           | 88.35%              | 88.89%                    |      | 88.68%                      |

## 模型预测

参考MindCV微调教程中[可视化模型推理结果](https://mindspore-lab.github.io/mindcv/zh/tutorials/finetune/#_12)小节，或是在validate.py中加入如下代码生成储存测试集真实值与预测值的文本文件./ckpt/pred.txt：

```python
# ... after model.eval()

# predited label
pred = np.argmax(model.predict(images).asnumpy(), axis=1)

# real label
images, labels = next(loader_eval.create_tuple_iterator())

# write pred.txt
prediction = np.array([pred, labels]).transpose()
np.savetxt("./ckpt/pred.txt", prediction, fmt="%s", header="pred \t real")
```

## 附录

以下表格展示了使用MindCV在多个CNN模型上对Aircraft数据集进行全量微调的精度（Top 1%）对比信息，该数据集上可实现的分类精度参见[Aircraft leaderboard](https://fgvc.org/leaderboard/Aircraft.html)和[paperwithcode网站](https://paperswithcode.com/sota/fine-grained-image-classification-on-fgvc)。

| 模型               | MindCV全量微调精度 | 参考精度                                                     |
| ------------------ | ------------------ | ------------------------------------------------------------ |
| mobilenet v3-small | 88.35%             | -                                                            |
| mobilenet v3-large | 92.22%             | [83.8%](https://opus.lib.uts.edu.au/handle/10453/156186)     |
| convnext-tiny      | 93.69%             | [84.23%](http://ise.thss.tsinghua.edu.cn/~mlong/doc/hub-pathway-transfer-learning-nips22.pdf) |
| resnest50          | 86.82%             | -                                                            |
