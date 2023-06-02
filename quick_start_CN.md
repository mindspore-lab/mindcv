# 快速入门

[![下载Notebook](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.8/resource/_static/logo_notebook.png)](https://download.mindspore.cn/toolkits/mindcv/tutorials/quick_start_CN.ipynb)&emsp;


[MindCV](https://github.com/mindspore-lab/mindcv)是一个基于 [MindSpore](https://www.mindspore.cn/) 开发的，致力于计算机视觉相关技术研发的开源工具箱。它提供大量的计算机视觉领域的经典模型和SoTA模型以及它们的预训练权重。同时，还提供了AutoAugment等SoTA算法来提高性能。通过解耦的模块设计，您可以轻松地将MindCV应用到您自己的CV任务中。本教程中我们将提供一个快速上手MindCV的指南。

本教程将以DenseNet分类模型为例，实现对Cifar10数据集的迁移学习，并在此流程中对MindCV各模块的用法作讲解。



## 环境准备

### 安装MindCV


```python
# install MindCV from git repo
!pip install git+https://github.com/mindspore-lab/mindcv.git
```

    Looking in indexes: http://100.125.0.87:32021/repository/pypi/simple
    Collecting git+https://github.com/mindspore-lab/mindcv.git
      Cloning https://github.com/mindspore-lab/mindcv.git to /tmp/pip-req-build-qnvkj8tb
      Running command git clone --filter=blob:none --quiet https://github.com/mindspore-lab/mindcv.git /tmp/pip-req-build-qnvkj8tb
      Resolved https://github.com/mindspore-lab/mindcv.git to commit 858fb89d5ee219be9e9ded86aaa15df06e9c9df5
      Preparing metadata (setup.py) ... [?25ldone
    [?25hRequirement already satisfied: numpy>=1.17.0 in /home/ma-user/anaconda3/envs/MindSpore/lib/python3.7/site-packages (from mindcv==0.0.2a0) (1.21.2)
    Requirement already satisfied: PyYAML>=5.3 in /home/ma-user/anaconda3/envs/MindSpore/lib/python3.7/site-packages (from mindcv==0.0.2a0) (5.3.1)
    Requirement already satisfied: tqdm in /home/ma-user/anaconda3/envs/MindSpore/lib/python3.7/site-packages (from mindcv==0.0.2a0) (4.46.1)
    Building wheels for collected packages: mindcv
      Building wheel for mindcv (setup.py) ... [?25ldone
    [?25h  Created wheel for mindcv: filename=mindcv-0.0.2a0-py3-none-any.whl size=165032 sha256=4e8c1f44ded45364658c6aa78f5e25025ba0cae023b33b402c6bdf4266983aa7
      Stored in directory: /tmp/pip-ephem-wheel-cache-q0tczanu/wheels/a8/17/96/9462c098d9c01564ef506e6666cb48246599c644a849c6aa62
    Successfully built mindcv
    Installing collected packages: mindcv
    Successfully installed mindcv-0.0.2a0


> 以下教程假设依赖包均已安装，若遇到依赖问题，请按照Git repo上的[安装指南](https://github.com/mindspore-lab/mindcv#dependency)进行安装

## 数据集读取

通过[mindcv.data](https://mindcv.readthedocs.io/en/latest/api/mindcv.data.html)中的[create_dataset](https://mindcv.readthedocs.io/en/latest/api/mindcv.data.html#mindcv.data.create_dataset)模块，我们可快速地读取标准数据集或自定义的数据集。


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

    170052608B [01:13, 2328662.39B/s]


[create_dataset](https://mindcv.readthedocs.io/en/latest/api/mindcv.data.html#mindcv.data.create_dataset)参数说明:

- name: 数据集名称，如mnist、cifar10、imagenet、' '表示自定义数据集。默认值：' '。

- dataset_dir: 包含数据集文件的根目录路径。默认：‘./’。

- split: ' '或拆分名称字符串（train/val/test），如果是' '，则不使用拆分。否则，它是根目录的子文件夹，例如train、val、test。默认值：'train'。

- shuffle: 是否混洗数据集。默认值：True。

- num_sample：获取的样本数。默认值：None，获取采样到的所有样本。

- num_shards：数据集分片数量。默认：None。如果指定此参数，num_samples将反映每个碎片的最大样本数。

- shard_id：当前分片的分片ID，默认：None。仅当同时指定num_shards时，才能指定此参数。

- num_parallel_workers: 指定读取数据的工作线程数。默认值：None。

- download: 是否下载数据集。默认值：False。

- num_aug_repeats: 重复增强数据集的重复次数。如果为0或1，则禁用重复增强。否则，将启用重复增强，常用选项为3。默认值：0。


## 数据处理及加载
1. 通过[create_transforms](https://mindcv.readthedocs.io/en/latest/api/mindcv.data.html#mindcv.data.create_transforms)函数, 可直接得到标准数据集合适的数据处理增强策略(transform list)，包括Cifar10, ImageNet上常用的数据处理策略。


```python
# 创建所需的数据增强操作的列表
trans = create_transforms(dataset_name='cifar10', image_resize=224)
```

[create_transforms](https://mindcv.readthedocs.io/en/latest/api/mindcv.data.html#mindcv.data.create_transforms)参数说明:

- dataset_name: 数据集名称。如果为“ ”，则为自定义数据集。当前应用与ImageNet相同的数据转换。如果给定标准数据集名称，包括imagenet、cifar10、mnist，则将返回预设转换。默认值：“ ”。

- image_resize：调整适应网络的图像大小。默认值：224。

- is_training：如果为True，则将在支持时应用增强。默认值：False。

- **kwargs： 额外其他参数。

2. 通过[mindcv.data.create_loader](https://mindcv.readthedocs.io/en/latest/api/mindcv.data.html#mindcv.data.create_loader)函数，进行数据转换和batch切分加载，我们需要将[create_transforms](https://mindcv.readthedocs.io/en/latest/api/mindcv.data.html#mindcv.data.create_transforms)返回的transform_list传入。


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

[create_loader](https://mindcv.readthedocs.io/en/latest/api/mindcv.data.html#mindcv.data.create_loader)参数说明:

- dataset: 通过标准数据集接口（mindspore.dataset.Cifar10Dataset，mindspore.dataset.CocoDataset）或者自定义数据集接口（mindspore.dataset.GeneratorDataset）加载过的数据集。

- batch_size: 指定每个批处理数据包含的数据条目。

- drop_remainder：确定是否删除小于批大小的数据最后一个块（默认值=False）。如果为True，并且如果有少于batch_size的数据可用于生成最后一个批处理，则这些数据将被删除，不会传播到子节点。

- is_training: 读取数据集的训练集（True）或验证集（False）。默认值：False。

- mixup：如果大于0，mixup将被启用（默认值：0.0）。

- cutmix：如果大于0，将启用cutmix（默认值：0.0）。此操作是实验性的。

- cutmix_prob: 为图像执行cutmix的概率（默认值：0.0）。

- num_classes: 分类的类别数。默认值：1000。

- transform: 将应用于图像的转换列表，由create_transform获得。如果为None，则将应用评估的默认imagenet转换。默认值：None。

- target_transform: 将应用于标签的转换列表。如果为None，则标签将转换为ms.int32类型。默认值：None。

- num_parallel_workers: 指定读取数据的工作线程数。默认值：None。

- python_multiprocessing；使用多个工作进程并行化Python操作。如果Python操作计算量很大（默认值为False），则此选项可能会很有用。


> 在notebook中避免重复执行[create_loader](https://mindcv.readthedocs.io/en/latest/api/mindcv.data.html#mindcv.data.create_loader)单个Cell，或在执行[create_dataset](https://mindcv.readthedocs.io/en/latest/api/mindcv.data.html#mindcv.data.create_dataset)之后再次执行。

## 模型创建和加载

使用[create_model](https://mindcv.readthedocs.io/en/latest/api/mindcv.models.html#mindcv.models.create_model)接口获得实例化的DenseNet，并加载预训练权重densenet_121_224.ckpt（ImageNet数据集训练得到）。




```python
from mindcv.models import create_model

# 实例化 DenseNet-121 模型并加载预训练权重。
network = create_model(model_name='densenet121', num_classes=num_classes, pretrained=True)
```

    32293888B [00:01, 28754770.92B/s]
    [WARNING] ME(1769:281472959711936,MainProcess):2022-12-21-16:03:22.690.392 [mindspore/train/serialization.py:712] For 'load_param_into_net', 2 parameters in the 'net' are not loaded, because they are not in the 'parameter_dict', please check whether the network structure is consistent when training and loading checkpoint.
    [WARNING] ME(1769:281472959711936,MainProcess):2022-12-21-16:03:22.691.960 [mindspore/train/serialization.py:714] classifier.weight is not loaded.
    [WARNING] ME(1769:281472959711936,MainProcess):2022-12-21-16:03:22.692.908 [mindspore/train/serialization.py:714] classifier.bias is not loaded.


> 由于Cifar10和ImageNet数据集所需类别数量不同，分类器参数无法共享，出现分类器参数无法加载的告警不影响微调。

[create_model](https://mindcv.readthedocs.io/en/latest/api/mindcv.models.html#mindcv.models.create_model)参数说明:

- model_name：需要加载的模型的规格的名称。

- num_classes：分类的类别数。默认值：1000。

- pretrained：是否加载与训练权重。默认值：False。

- in_channels：输入通道。默认值：3。

- checkpoint_path：checkpoint的路径。默认值：“ ”。

- ema：是否使用ema方法 默认值: False。

使用[mindcv.loss.create_loss](https://mindcv.readthedocs.io/en/latest/api/mindcv.loss.html#mindcv.loss.create_loss)接口创建损失函数（cross_entropy loss）。

## 模型训练

通过[create_loss](https://mindcv.readthedocs.io/en/latest/api/mindcv.loss.html#mindcv.loss.create_loss)接口获得损失函数


```python
from mindcv.loss import create_loss

loss = create_loss(name='CE')
```

[create_loss](https://mindcv.readthedocs.io/en/latest/api/mindcv.loss.html#mindcv.loss.create_loss)参数说明：

- name：损失函数名称：'CE'用于交叉熵损失。'BCE'：二进制交叉熵。默认值：“CE”。

- weight：指定各类别的权重。数据类型仅支持float32或float16。默认值: None。

- reduction：指定应用于输出结果的计算方式，比如’none’、’mean’、’sum’，默认值：’mean’。

- label_smoothing：标签平滑值，用于计算Loss时防止模型过拟合的正则化手段。取值范围为[0.0, 1.0]。默认值：0.0。

- aux_factor：辅助损耗因数。如果模型具有辅助逻辑输出（即深度监控），如inception_v3模型，则设置aux_fuactor>0.0。默认值：0.0。

使用[create_scheduler](https://mindcv.readthedocs.io/en/latest/api/mindcv.scheduler.html#mindcv.scheduler.create_scheduler)接口设置学习率策略。


```python
from mindcv.scheduler import create_scheduler

# 设置学习率策略
lr_scheduler = create_scheduler(steps_per_epoch=num_batches,
                                scheduler='constant',
                                lr=0.0001)
```

[create_scheduler](https://mindcv.readthedocs.io/en/latest/api/mindcv.scheduler.html#mindcv.scheduler.create_scheduler)参数说明:

- steps_pre_epoch：完成一轮训练所需要的步数。

- scheduler：学习率策略的名称。默认值：‘constant’。

- lr：学习率。默认值：0.01。

- min_lr：decay时学习率的最小值。默认值：1e-6。

- warmup_epochs：如果学习率策略支持，用来预热学习率。默认值：3。

- warmup_factor：学习率策略的预热阶段是一个线性增加的学习率，开始因子是warmup_factor，即第一个step/epoch的学习率是lr * warmup_actor，而预热阶段的结束学习率为lr。默认值：0.0。

- decay_epochs：对于“cosine_decay”学习率策略，在decay_epochs中将学习率衰减到min_lr。对于“step_decay”学习率策略，每decay_epochs将学习率衰减一个decay_rate因子。默认值：10。

- decay_rate：学习率衰减因子。默认值：0.9。

- milestones：“multi_step_decay”学习率策略的列表。

- num_epochs：训练epoch的数量。

- lr_epoch_stair：如果为True，则学习率将在每个epoch的开始时更新，并且学习率将在一个epoch中对每个批次保持一致。否则，学习率将在每个步骤中动态更新。（默认值=False）


使用[create_optimizer](https://mindcv.readthedocs.io/en/latest/api/mindcv.optim.html#mindcv.optim.create_optimizer)接口创建优化器。


```python
from mindcv.optim import create_optimizer

# 设置优化器
opt = create_optimizer(network.trainable_params(), opt='adam', lr=lr_scheduler)
```

[create_optimizer](https://mindcv.readthedocs.io/en/latest/api/mindcv.optim.html#mindcv.optim.create_optimizer)参数说明:

- params：需要优化的参数的列表。

- opt：优化器。默认值：'adam'。

- lr：学习率的最大值。默认值：1e-3。

- weight_decay：权重衰减系数。默认值：0。

- momentum：如果优化器支持，则会产生动量。默认值：0.9。

- nesterov：是否使用Nesterov加速梯度（NAG）算法更新梯度。默认值：False。

- filter_bias_and_bn：是否过滤批次规范参数和权重衰减的偏差。如果为True，权重衰减将不适用于Conv层或Dense层中的BN参数和bias。默认值：True。

- loss_scale：损失函数值缩放比例。默认值：1.0。

- checkpoint_path：优化器checkpoint路径。

- eps：添加到分母以提高数值稳定性的项。默认值：1e-10。

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

> 详细教程请参考 [使用yaml文件的教程](https://mindcv.readthedocs.io/en/latest/tutorials/learn_about_config.html)




```python
!git clone https://github.com/mindspore-lab/mindcv.git
!cd mindcv
```


```python
#  单卡训练
!python train.py -c configs/squeezenet/squeezenet_1.0_gpu.yaml --data_dir /path/to/dataset --distribute False
```


```python
!python validate.py -c configs/squeezenet/squeezenet_1.0_gpu.yaml --data_dir /path/to/dataset --ckpt_path /path/to/ckpt
```
