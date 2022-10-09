# 快速入门

本教程中我们将提供一个快速上手`mindcv`的指南。

本教程将以DenseNet分类模型为例，实现对Cifar10数据集的迁移学习，并在此流程中对MindCV各模块的用法作讲解。



## 环境准备

### 安装MindCV


```python
# instal mindcv from git repo
!pip install git+https://github.com/mindlab-ai/mindcv.git
```

    Looking in indexes: http://mirrors.aliyun.com/pypi/simple
    Collecting git+https://github.com/mindlab-ai/mindcv.git
      Cloning https://github.com/mindlab-ai/mindcv.git to /tmp/pip-req-build-2t1sum4n
      Running command git clone --filter=blob:none --quiet https://github.com/mindlab-ai/mindcv.git /tmp/pip-req-build-2t1sum4n
      Resolved https://github.com/mindlab-ai/mindcv.git to commit 81fa3df8a7292c03b2a69b1456dadcfbe7ae9b9c
      Preparing metadata (setup.py) ... [?25ldone
    [?25hRequirement already satisfied: numpy>=1.17.0 in /opt/conda/envs/xgraph/lib/python3.8/site-packages (from mindcv==0.0.1) (1.22.4)
    Requirement already satisfied: PyYAML>=5.3 in /opt/conda/envs/xgraph/lib/python3.8/site-packages (from mindcv==0.0.1) (5.4)
    Requirement already satisfied: tqdm in /opt/conda/envs/xgraph/lib/python3.8/site-packages (from mindcv==0.0.1) (4.59.0)
    [33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m[33m
    [0m

> 以下教程假设依赖包均已安装，若遇到依赖问题，请按照Git repo上的[安装指南](https://github.com/mindlab-ai/mindcv#dependency)进行安装

## 数据集读取

通过`mindcv.data`中的`create_dataset`模块，我们可快速地读取标准数据集或自定义的数据集。


```python
from mindcv.data import create_dataset, create_transforms, create_loader
import os

# 数据集路径
cifar10_dir = './datasets/cifar/cifar-10-batches-bin' # 你的数据存放路径
num_classes = 10 # 类别数
num_workers = 8 # 数据读取及加载的工作线程数 
download = not os.path.exists(cifar10_dir)

# 创建数据集
dataset_train = create_dataset(name='cifar10', root=cifar10_dir, split='train', shuffle=True, num_parallel_workers=num_workers, download=download)
```

`create_dataset`参数说明:

- name: 数据集名称

- dataset_dir: 包含数据集文件的根目录路径。

- split: 读取数据集的训练集（"train"）或验证集（"val"）。默认值："train"。

- shuffle: 是否混洗数据集。默认值：None。

- num_parallel_workers: 指定读取数据的工作线程数。默认值：None。

- download: 是否下载数据集。默认值：False。


## 数据处理及加载
1. 通过`create_transforms`函数, 可直接得到标准数据集合适的数据处理增强策略(transform list)，包括Cifar10, imagenet上常用的数据处理策略。


```python
# 创建所需的数据增强操作的列表
trans = create_transforms(dataset_name='cifar10', image_resize=224)
```

`create_transforms`参数说明:

- name: 数据集名称

- dataset_dir: 包含数据集文件的根目录路径。

- split: 读取数据集的训练集（"train"）或验证集（"val"）。默认值："train"。

- shuffle: 是否混洗数据集。默认值：None。

- num_parallel_workers: 指定读取数据的工作线程数。默认值：None。

- download: 是否下载数据集。默认值：False。

2. 通过`mindcv.data.create_loader`函数，进行数据转换和batch切分加载，我们需要将`create_transform`返回的transform_list传入。


```python
# 执行数据增强操作，生成所需数据集。
loader_train = create_loader(dataset=dataset_train,
                             batch_size=64,
                             is_training=True,
                             num_classes=num_classes,
                             transform=trans,
                             num_parallel_workers=num_workers)

num_batches = loader_train.get_dataset_size()
```

`create_loader`参数说明:

- dataset: 通过标准数据集接口（mindspore.dataset.Cifar10Dataset，mindspore.dataset.CocoDataset）或者自定义数据集接口（mindspore.dataset.GeneratorDataset）加载过的数据集。

- batch_size: 指定每个批处理数据包含的数据条目。

- is_training: 读取数据集的训练集（True）或验证集（False）。默认值：False。

- num_classes: 分类的类别数。默认值：1000。
    
- transform: 所需的数据增强操作的列表。默认值：None。

- num_parallel_workers: 指定读取数据的工作线程数。默认值：None。


> 在notebook中避免重复执行`create_loader`单个Cell，或在执行`create_dataset`之后再次执行

## 模型创建和加载

使用`create_model`接口获得实例化的DenseNet，并加载预训练权重densenet_121_imagenet2012.ckpt（ImageNet数据集训练得到）。




```python
from mindcv.models import create_model

# 实例化 DenseNet-121 模型并加载预训练权重。
network = create_model(model_name='densenet121', num_classes=num_classes, pretrained=True)
```

    [WARNING] ME(57165:140402355906368,MainProcess):2022-09-22-09:08:11.784.095 [mindspore/train/serialization.py:709] For 'load_param_into_net', 2 parameters in the 'net' are not loaded, because they are not in the 'parameter_dict', please check whether the network structure is consistent when training and loading checkpoint.
    [WARNING] ME(57165:140402355906368,MainProcess):2022-09-22-09:08:11.785.187 [mindspore/train/serialization.py:714] classifier.weight is not loaded.
    [WARNING] ME(57165:140402355906368,MainProcess):2022-09-22-09:08:11.785.831 [mindspore/train/serialization.py:714] classifier.bias is not loaded.
    

> 由于Cifar10和ImageNet数据集所需类别数量不同，分类器参数无法共享，出现分类器参数无法加载的告警不影响微调。

`create_model`参数说明:

- model_name: 需要加载的模型的规格的名称。

- num_classes: 分类的类别数。默认值：1000。

- pretrained: 是否加载与训练权重。默认值：False。

使用`mindcv.loss.create_loss`接口创建损失函数（cross_entropy loss）。

## 模型训练

通过`create_loss`接口获得损失函数


```python
from mindcv.loss import create_loss

loss = create_loss(name='CE')
```

使用`create_scheduler`接口设置学习率策略（warmup_consine_decay）。


```python
from mindcv.scheduler import create_scheduler

# 设置学习率策略
lr_scheduler = create_scheduler(steps_per_epoch=num_batches,
                                scheduler='constant',
                                lr=0.0001)
```

参数说明:

- steps_pre_epoch: 完成一轮训练所需要的步数。

- scheduler: 学习率策略的名称。

- lr: 学习率。

- min_lr: decay时学习率的最小值。

使用`create_optimizer`接口创建优化器。


```python
from mindcv.optim import create_optimizer

# 设置优化器
opt = create_optimizer(network.trainable_params(), opt='adam', lr=lr_scheduler) 
```

参数说明:

- params: 需要优化的参数的列表。

- scheduler: 学习了策略的名称。

- lr: 学习率的最大值。

- min_lr: 学习率的最小值。


使用[mindspore.Model](https://mindspore.cn/docs/zh-CN/r1.8/api_python/mindspore/mindspore.Model.html)接口根据用户传入的参数封装可训练的实例。


```python
from mindspore import Model

# 封装可训练或推理的实例
model = Model(network, loss_fn=loss, optimizer=opt, metrics={'accuracy'})
```

使用[`mindspore.Model.train`](https://mindspore.cn/docs/zh-CN/r1.8/api_python/mindspore/mindspore.Model.html#mindspore.Model.train)接口进行模型训练。


```python
from mindspore import LossMonitor, TimeMonitor, CheckpointConfig, ModelCheckpoint

# 设置在训练过程中保存网络参数的回调函数
ckpt_save_dir = './ckpt' 
ckpt_config = CheckpointConfig(save_checkpoint_steps=num_batches)
ckpt_cb = ModelCheckpoint(prefix='densenet121-cifar10',
                          directory=ckpt_save_dir,
                          config=ckpt_config)

model.train(5, loader_train, callbacks=[LossMonitor(num_batches//5), TimeMonitor(num_batches//5), ckpt_cb], dataset_sink_mode=False)
```

    epoch: 1 step: 156, loss is 0.36890333890914917
    epoch: 1 step: 312, loss is 0.2963641285896301
    epoch: 1 step: 468, loss is 0.08654475212097168
    epoch: 1 step: 624, loss is 0.1908271610736847
    epoch: 1 step: 780, loss is 0.1770080029964447
    Train epoch time: 262146.330 ms, per step time: 335.225 ms
    epoch: 2 step: 154, loss is 0.04639885947108269
    epoch: 2 step: 310, loss is 0.12687519192695618
    epoch: 2 step: 466, loss is 0.03369298204779625
    epoch: 2 step: 622, loss is 0.12257681041955948
    epoch: 2 step: 778, loss is 0.13823091983795166
    Train epoch time: 237231.079 ms, per step time: 303.365 ms
    epoch: 3 step: 152, loss is 0.03291231021285057
    epoch: 3 step: 308, loss is 0.04826178774237633
    epoch: 3 step: 464, loss is 0.06561325490474701
    epoch: 3 step: 620, loss is 0.028005748987197876
    epoch: 3 step: 776, loss is 0.14322009682655334
    Train epoch time: 240640.121 ms, per step time: 307.724 ms
    epoch: 4 step: 150, loss is 0.04635673016309738
    epoch: 4 step: 306, loss is 0.006769780069589615
    epoch: 4 step: 462, loss is 0.07550926506519318
    epoch: 4 step: 618, loss is 0.007201619446277618
    epoch: 4 step: 774, loss is 0.02128467708826065
    Train epoch time: 244391.659 ms, per step time: 312.521 ms
    epoch: 5 step: 148, loss is 0.00641212984919548
    epoch: 5 step: 304, loss is 0.013159077614545822
    epoch: 5 step: 460, loss is 0.021671295166015625
    epoch: 5 step: 616, loss is 0.01827814429998398
    epoch: 5 step: 772, loss is 0.008501190692186356
    Train epoch time: 240139.144 ms, per step time: 307.083 ms
    


```python
# 加载验证数据集
dataset_val = create_dataset(name='cifar10', root=cifar10_dir, split='test', shuffle=True, num_parallel_workers=num_workers, download=download)

# 执行数据增强操作，生成所需数据集。
loader_val = create_loader(dataset=dataset_val,
                           batch_size=64,
                           is_training=False,
                           num_classes=num_classes,
                           transform=trans,
                           num_parallel_workers=num_workers)
```

加载微调后的参数文件（densenet-cifar10-10_782.ckpt）到模型。

根据用户传入的参数封装可推理的实例，加载验证数据集，验证微调的 DenseNet121模型精度。


```python
# 验证微调后的DenseNet-121的精度
acc = model.eval(loader_val, dataset_sink_mode=False)
print(acc)
```

    {'accuracy': 0.9577}
    

## 使用YAML文件进行模型训练和验证

我们还可以直接使用设置好模型参数的yaml文件，通过`train.py`和`validate.py`脚本来快速来对模型进行训练和验证。以下是在ImageNet上训练SqueezenetV1的示例 （需要将imagenet提前下载到目录下）

> 详细教程请参考 [使用yaml文件的教程](./learn_about_config.ipynb)




```python
!git clone https://github.com/mindlab-ai/mindcv.git
!cd mindcv
```


```python
!python train.py -c configs/squeezenet/squeezenet_1.0_gpu.yaml 
```


```python
!python validate.py -c configs/squeezenet/squeezenet_1.0_gpu.yaml 
```
