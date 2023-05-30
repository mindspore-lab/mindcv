# 快速入门

[![下载Notebook](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.8/resource/_static/logo_notebook.png)](https://download.mindspore.cn/toolkits/mindcv/tutorials/quick_start_CN.ipynb)


[MindCV](https://github.com/mindspore-lab/mindcv)是一个基于[MindSpore]开发的，致力于计算机视觉相关技术研发的开源工具箱。
它提供大量的计算机视觉领域的经典模型和SoTA模型以及它们的预训练权重。同时，还提供了AutoAugment等SoTA算法来提高性能。
通过解耦的模块设计，您可以轻松地将MindCV应用到您自己的CV任务中。本教程中我们将提供一个快速上手MindCV的指南。

本教程将以DenseNet分类模型为例，实现对[CIFAR-10]数据集的迁移学习，并在此流程中对MindCV各模块的用法作讲解。

[MindCV]: https://github.com/mindspore-lab/mindcv
[MindSpore]: https://www.mindspore.cn

## 环境准备

详见[安装](../installation.md)。

## 数据

### 数据集

通过[mindcv.data](https://mindcv.readthedocs.io/en/latest/api/mindcv.data.html)中的[create_dataset](https://mindcv.readthedocs.io/en/latest/api/mindcv.data.html#mindcv.data.create_dataset)模块，我们可快速地读取标准数据集或自定义的数据集。

```python
import os
from mindcv.data import create_dataset, create_transforms, create_loader

cifar10_dir = './datasets/cifar/cifar-10-batches-bin'  # 你的数据存放路径
num_classes = 10  # 类别数
num_workers = 8  # 数据读取及加载的工作线程数

# 创建数据集
dataset_train = create_dataset(
    name='cifar10', root=cifar10_dir, split='train', shuffle=True, num_parallel_workers=num_workers
)
```

### 数据变换

通过[create_transforms](https://mindcv.readthedocs.io/en/latest/api/mindcv.data.html#mindcv.data.create_transforms)函数, 可直接得到标准数据集合适的数据处理增强策略(transform list)，包括Cifar10, ImageNet上常用的数据处理策略。

```python
# 创建所需的数据增强操作的列表
trans = create_transforms(dataset_name='cifar10', image_resize=224)
```

### 数据加载

通过[mindcv.data.create_loader](https://mindcv.readthedocs.io/en/latest/api/mindcv.data.html#mindcv.data.create_loader)函数，进行数据转换和batch切分加载，我们需要将[create_transforms](https://mindcv.readthedocs.io/en/latest/api/mindcv.data.html#mindcv.data.create_transforms)返回的transform_list传入。

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

> 在notebook中避免重复执行[create_loader](https://mindcv.readthedocs.io/en/latest/api/mindcv.data.html#mindcv.data.create_loader)单个Cell，或在执行[create_dataset](https://mindcv.readthedocs.io/en/latest/api/mindcv.data.html#mindcv.data.create_dataset)之后再次执行。

## 模型创建和加载

使用[create_model](https://mindcv.readthedocs.io/en/latest/api/mindcv.models.html#mindcv.models.create_model)接口获得实例化的DenseNet，并加载预训练权重densenet_121_224.ckpt（ImageNet数据集训练得到）。

```python
from mindcv.models import create_model

# 实例化 DenseNet-121 模型并加载预训练权重。
network = create_model(model_name='densenet121', num_classes=num_classes, pretrained=True)
```

> 由于CIFAR-10和ImageNet数据集所需类别数量不同，分类器参数无法共享，出现分类器参数无法加载的告警不影响微调。

## 损失函数

通过[create_loss](https://mindcv.readthedocs.io/en/latest/api/mindcv.loss.html#mindcv.loss.create_loss)接口获得损失函数

```python
from mindcv.loss import create_loss

loss = create_loss(name='CE')
```

## 学习率调度器

使用[create_scheduler](https://mindcv.readthedocs.io/en/latest/api/mindcv.scheduler.html#mindcv.scheduler.create_scheduler)接口设置学习率策略。

```python
from mindcv.scheduler import create_scheduler

# 设置学习率策略
lr_scheduler = create_scheduler(steps_per_epoch=num_batches,
                                scheduler='constant',
                                lr=0.0001)
```

## 优化器

使用[create_optimizer](https://mindcv.readthedocs.io/en/latest/api/mindcv.optim.html#mindcv.optim.create_optimizer)接口创建优化器。

```python
from mindcv.optim import create_optimizer

# 设置优化器
opt = create_optimizer(network.trainable_params(), opt='adam', lr=lr_scheduler)
```

## 训练

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

    [WARNING] DEVICE(1769,ffff87c70ac0,python):2022-12-21-16:04:30.001.890 [mindspore/ccsrc/plugin/device/ascend/hal/device/kernel_select_ascend.cc:330] FilterRaisedOrReducePrecisionMatchedKernelInfo] Operator:[Default/network-WithLossCell/_loss_fn-CrossEntropySmooth/GatherD-op5273] don't support int64, reduce precision from int64 to int32.


    epoch: 1 step: 156, loss is 2.0816354751586914
    epoch: 1 step: 312, loss is 1.4474115371704102
    epoch: 1 step: 468, loss is 0.8935483694076538
    epoch: 1 step: 624, loss is 0.5588696002960205
    epoch: 1 step: 780, loss is 0.3161369860172272


    [WARNING] DEVICE(1769,ffff87c70ac0,python):2022-12-21-16:09:20.261.851 [mindspore/ccsrc/plugin/device/ascend/hal/device/kernel_select_ascend.cc:330] FilterRaisedOrReducePrecisionMatchedKernelInfo] Operator:[Default/network-WithLossCell/_loss_fn-CrossEntropySmooth/GatherD-op16720] don't support int64, reduce precision from int64 to int32.


    Train epoch time: 416429.509 ms, per step time: 532.519 ms
    epoch: 2 step: 154, loss is 0.19752007722854614
    epoch: 2 step: 310, loss is 0.14635677635669708
    epoch: 2 step: 466, loss is 0.3511860966682434
    epoch: 2 step: 622, loss is 0.12542471289634705
    epoch: 2 step: 778, loss is 0.22351759672164917
    Train epoch time: 156746.872 ms, per step time: 200.444 ms
    epoch: 3 step: 152, loss is 0.08965137600898743
    epoch: 3 step: 308, loss is 0.22765043377876282
    epoch: 3 step: 464, loss is 0.19035443663597107
    epoch: 3 step: 620, loss is 0.06591956317424774
    epoch: 3 step: 776, loss is 0.0934530645608902
    Train epoch time: 156574.210 ms, per step time: 200.223 ms
    epoch: 4 step: 150, loss is 0.03782692924141884
    epoch: 4 step: 306, loss is 0.023876197636127472
    epoch: 4 step: 462, loss is 0.038690414279699326
    epoch: 4 step: 618, loss is 0.15388774871826172
    epoch: 4 step: 774, loss is 0.1581358164548874
    Train epoch time: 158398.108 ms, per step time: 202.555 ms
    epoch: 5 step: 148, loss is 0.06556802988052368
    epoch: 5 step: 304, loss is 0.006707251071929932
    epoch: 5 step: 460, loss is 0.02353120595216751
    epoch: 5 step: 616, loss is 0.014183484017848969
    epoch: 5 step: 772, loss is 0.09367241710424423
    Train epoch time: 154978.618 ms, per step time: 198.182 ms


## 评估

现在让我们在[CIFAR-10]上对刚刚训练的模型进行评估。

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

加载微调后的参数文件（densenet121-cifar10-5_782.ckpt）到模型。

根据用户传入的参数封装可推理的实例，加载验证数据集，验证微调的 DenseNet121模型精度。

```python
# 验证微调后的DenseNet121的精度
acc = model.eval(loader_val, dataset_sink_mode=False)
print(acc)
```

    [WARNING] DEVICE(1769,ffff87c70ac0,python):2022-12-21-16:24:11.927.472 [mindspore/ccsrc/plugin/device/ascend/hal/device/kernel_select_ascend.cc:330] FilterRaisedOrReducePrecisionMatchedKernelInfo] Operator:[Default/network-WithLossCell/_loss_fn-CrossEntropySmooth/GatherD-op24314] don't support int64, reduce precision from int64 to int32.


    {'accuracy': 0.951}


    [WARNING] DEVICE(1769,ffff87c70ac0,python):2022-12-21-16:25:01.871.273 [mindspore/ccsrc/plugin/device/ascend/hal/device/kernel_select_ascend.cc:330] FilterRaisedOrReducePrecisionMatchedKernelInfo] Operator:[Default/network-WithLossCell/_loss_fn-CrossEntropySmooth/GatherD-op27139] don't support int64, reduce precision from int64 to int32.


## 使用YAML文件进行模型训练和验证

我们还可以直接使用设置好模型参数的yaml文件，通过`train.py`和`validate.py`脚本来快速来对模型进行训练和验证。以下是在[ImageNet](https://www.image-net.org/challenges/LSVRC/2012/index.php)上训练SqueezenetV1的示例 （需要将ImageNet提前下载到目录下）

> 详细教程请参考 [使用yaml文件的教程](./configuration.md)

```shell
#  单卡训练
python train.py -c configs/squeezenet/squeezenet_1.0_gpu.yaml --data_dir /path/to/dataset --distribute False
```

```shell
python validate.py -c configs/squeezenet/squeezenet_1.0_gpu.yaml --data_dir /path/to/dataset --ckpt_path /path/to/ckpt
```

[CIFAR-10]: https://www.cs.toronto.edu/~kriz/cifar.html
[ImageNet]: https://www.image-net.org/challenges/LSVRC/2012/index.php
