# Quick Start

[![Download Notebook](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.8/resource/_static/logo_notebook_en.png)](https://download.mindspore.cn/toolkits/mindcv/tutorials/quick_start.ipynb)


[MindCV] is an open source toolbox for computer vision research and development based on [MindSpore].
It collects a series of classic and SoTA vision models, such as ResNet and SwinTransformer, along with their pretrained weights.
SoTA methods such as AutoAugment are also provided for performance improvement.
With the decoupled module design, it is easy to apply or adapt MindCV to your own CV tasks.
In this tutorial, we will provide a quick start guideline for MindCV.

This tutorial will take DenseNet classification model as an example to implement transfer training on [CIFAR-10] dataset, and explain the usage of MindCV modules in this process.

[MindCV]: https://github.com/mindspore-lab/mindcv
[MindSpore]: https://www.mindspore.cn/en

## Environment Setting

See [Installation](../installation.md) for details.

## Data

### Dataset

Through the [create_dataset](https://mindcv.readthedocs.io/en/latest/api/mindcv.data.html#mindcv.data.create_dataset) module in [mindcv.data](https://mindcv.readthedocs.io/en/latest/api/mindcv.data.html), we can quickly load standard datasets or customized datasets.

```python
import os
from mindcv.data import create_dataset, create_transforms, create_loader

cifar10_dir = './datasets/cifar/cifar-10-batches-bin'  # your dataset path
num_classes = 10  # num of classes
num_workers = 8  # num of parallel workers

# create dataset
dataset_train = create_dataset(
    name='cifar10', root=cifar10_dir, split='train', shuffle=True, num_parallel_workers=num_workers
)
```

### Transform

Through the [create_transforms](https://mindcv.readthedocs.io/en/latest/api/mindcv.data.html#mindcv.data.create_transforms) function, you can directly obtain the appropriate data processing augmentation strategies (transform list) for standard datasets, including common data processing strategies on Cifar10 and Imagenet.

```python
# create transforms
trans = create_transforms(dataset_name='cifar10', image_resize=224)
```

### Loader

The [mindcv.data.create_loader](https://mindcv.readthedocs.io/en/latest/api/mindcv.data.html#mindcv.data.create_loader) function is used for data conversion and batch split loading. We need to pass in the transform_list returned by [create_transforms](https://mindcv.readthedocs.io/en/latest/api/mindcv.data.html#mindcv.data.create_transforms).

```python
# Perform data augmentation operations to generate the required dataset.
loader_train = create_loader(dataset=dataset_train,
                             batch_size=64,
                             is_training=True,
                             num_classes=num_classes,
                             transform=trans,
                             num_parallel_workers=num_workers)

num_batches = loader_train.get_dataset_size()
```

> Avoid repeatedly executing a single cell of [create_loader](https://mindcv.readthedocs.io/en/latest/api/mindcv.data.html#mindcv.data.create_loader) in notebook, or execute again after executing [create_dataset](https://mindcv.readthedocs.io/en/latest/api/mindcv.data.html#mindcv.data.create_dataset).

## Model

Use the [create_model](https://mindcv.readthedocs.io/en/latest/api/mindcv.models.html#mindcv.models.create_model) interface to obtain the instantiated DenseNet and load the pretraining weight(obtained from ImageNet dataset training).

```python
from mindcv.models import create_model

# instantiate the DenseNet121 model and load the pretraining weights.
network = create_model(model_name='densenet121', num_classes=num_classes, pretrained=True)
```

> Because the number of classes required by CIFAR-10 and ImageNet datasets is different, the classifier parameters cannot be shared, and the warning that the classifier parameters cannot be loaded does not affect the fine-tuning.

## Loss

By [create_loss](https://mindcv.readthedocs.io/en/latest/api/mindcv.loss.html#mindcv.loss.create_loss) interface obtains loss function.

```python
from mindcv.loss import create_loss

loss = create_loss(name='CE')
```

## Learning Rate Scheduler

Use [create_scheduler](https://mindcv.readthedocs.io/en/latest/api/mindcv.scheduler.html#mindcv.scheduler.create_scheduler) interface sets the learning rate scheduler.

```python
from mindcv.scheduler import create_scheduler

# learning rate scheduler
lr_scheduler = create_scheduler(steps_per_epoch=num_batches,
                                scheduler='constant',
                                lr=0.0001)
```

## Optimizer

Use [create_optimizer](https://mindcv.readthedocs.io/en/latest/api/mindcv.optim.html#mindcv.optim.create_optimizer) interface creates an optimizer.

```python
from mindcv.optim import create_optimizer

# create optimizer
opt = create_optimizer(network.trainable_params(), opt='adam', lr=lr_scheduler)
```

## Training

Use the [mindspore.Model](https://mindspore.cn/docs/zh-CN/r1.8/api_python/mindspore/mindspore.Model.html) interface to encapsulate trainable instances according to the parameters passed in by the user.

```python
from mindspore import Model

# Encapsulates examples that can be trained or inferred
model = Model(network, loss_fn=loss, optimizer=opt, metrics={'accuracy'})
```

Use the [`mindspore.Model.train`](https://mindspore.cn/docs/zh-CN/r1.8/api_python/mindspore/mindspore.Model.html#mindspore.Model.train) interface for model training.

```python
from mindspore import LossMonitor, TimeMonitor, CheckpointConfig, ModelCheckpoint

# Set the callback function for saving network parameters during training.
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


## Evaluation

Now, let's evaluate the trained model on the validation set of [CIFAR-10].

```python
# Load validation dataset
dataset_val = create_dataset(
    name='cifar10', root=cifar10_dir, split='test', shuffle=True, num_parallel_workers=num_workers
)

# Perform data enhancement operations to generate the required dataset.
loader_val = create_loader(dataset=dataset_val,
                           batch_size=64,
                           is_training=False,
                           num_classes=num_classes,
                           transform=trans,
                           num_parallel_workers=num_workers)
```

Load the fine-tuning parameter file (densenet121-cifar10-5_782.ckpt) to the model.

Encapsulate inferable instances according to the parameters passed in by the user, load the validation data set, and verify the precision of the fine-tuned DenseNet121 model.

```python
# Verify the accuracy of DenseNet121 after fine-tune
acc = model.eval(loader_val, dataset_sink_mode=False)
print(acc)
```

    [WARNING] DEVICE(1769,ffff87c70ac0,python):2022-12-21-16:24:11.927.472 [mindspore/ccsrc/plugin/device/ascend/hal/device/kernel_select_ascend.cc:330] FilterRaisedOrReducePrecisionMatchedKernelInfo] Operator:[Default/network-WithLossCell/_loss_fn-CrossEntropySmooth/GatherD-op24314] don't support int64, reduce precision from int64 to int32.


    {'accuracy': 0.951}


    [WARNING] DEVICE(1769,ffff87c70ac0,python):2022-12-21-16:25:01.871.273 [mindspore/ccsrc/plugin/device/ascend/hal/device/kernel_select_ascend.cc:330] FilterRaisedOrReducePrecisionMatchedKernelInfo] Operator:[Default/network-WithLossCell/_loss_fn-CrossEntropySmooth/GatherD-op27139] don't support int64, reduce precision from int64 to int32.


## Use YAML files for model training and validation

We can also use the yaml file with the model parameters set directly to quickly train and verify the model through `train.py` and `validate.py` scripts.
The following is an example of training SqueezenetV1 on [ImageNet] (you need to download ImageNet to the directory in advance).

> For detailed tutorials, please refer to the [tutorial](./configuration.md).

```shell
# standalone training on a CPU/GPU/Ascend device
python train.py -c configs/squeezenet/squeezenet_1.0_gpu.yaml --data_dir /path/to/dataset --distribute False
```

```shell
python validate.py -c configs/squeezenet/squeezenet_1.0_gpu.yaml --data_dir /path/to/dataset --ckpt_path /path/to/ckpt
```

[CIFAR-10]: https://www.cs.toronto.edu/~kriz/cifar.html
[ImageNet]: https://www.image-net.org/challenges/LSVRC/2012/index.php
