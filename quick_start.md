# Quick Start

[![下载Notebook](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.8/resource/_static/logo_notebook.png)](https://download.mindspore.cn/toolkits/mindcv/tutorials/quick_start.ipynb)&emsp;


[MindCV](https://github.com/mindspore-lab/mindcv) is an open source toolbox for computer vision research and development based on [MindSpore](https://www.mindspore.cn/en). It collects a series of classic and SoTA vision models, such as ResNet and SwinTransformer, along with their pretrained weights. SoTA methods such as AutoAugment are also provided for performance improvement. With the decoupled module design, it is easy to apply or adapt MindCV to your own CV tasks. In this tutorial, we will provide a quick start guideline for MindCV.

This tutorial will take DenseNet classification model as an example to implement migration training for Cifar10 dataset, and explain the usage of MindCV modules in this process.


## Environment Setting

### Installing MindCV


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


>The following tutorials assume that all dependent packages have been installed. If you encounter dependency problems, please follow the [installation guide](https://github.com/mindspore-lab/mindcv#dependency) on Git repo.


## Dataset Load

Through the [create_dataset](https://mindcv.readthedocs.io/en/latest/api/mindcv.data.html#mindcv.data.create_dataset)  module in [mindcv.data](https://mindcv.readthedocs.io/en/latest/api/mindcv.data.html), we can quickly load standard datasets or customized datasets.


```python
from mindcv.data import create_dataset, create_transforms, create_loader
import os

# dataset path
cifar10_dir = './datasets/cifar/cifar-10-batches-bin' # your dataset path
num_classes = 10 # num of classes
num_workers = 8 # Number of parallel workers
download = not os.path.exists(cifar10_dir)

# create dataset
dataset_train = create_dataset(name='cifar10', root=cifar10_dir, split='train', shuffle=True, num_parallel_workers=num_workers, download=download)
```

    170052608B [01:13, 2328662.39B/s]


[create_dataset](https://mindcv.readthedocs.io/en/latest/api/mindcv.data.html#mindcv.data.create_dataset) parameters:

- name: dataset name like mnist, cifar10, imagenet, ' ' means a customized dataset. Default: ' '.

- dataset_dir: dataset root dir. Default: './'.

- split: data split, ' ' or split name string (train/val/test), if it is ' ', no split is used. Otherwise, it is a subfolder of root dir, e.g., train, val, test. Default: ‘train’.

- shuffle: whether to shuffle the dataset. Default: True.

- num_sample: Number of elements to sample (default=None, which means sample all elements).

- num_shards: Number of shards that the dataset will be divided into (default=None). When this argument is specified, num_samples reflects the maximum sample number of per shard.

- shard_id: The shard ID within num_shards (default=None). This argument can only be specified when num_shards is also specified.

- num_parallel_workers: Number of workers to read the data (default=None, set in the config).

- download: whether to download the dataset. Default: False.

- num_aug_repeats: Number of dataset repeatition for repeated augmentation. If 0 or 1, repeated augmentation is diabled. Otherwise, repeated augmentation is enabled and the common choice is 3. (Default: 0)


## Data Processing and Loading

1. Through the [create_transforms](https://mindcv.readthedocs.io/en/latest/api/mindcv.data.html#mindcv.data.create_transforms) function, you can directly obtain the appropriate data processing augmentation strategies (transform list) for standard datasets, including common data processing strategies on Cifar10 and Imagenet.



```python
# create transforms
trans = create_transforms(dataset_name='cifar10', image_resize=224)
```

[create_transforms](https://mindcv.readthedocs.io/en/latest/api/mindcv.data.html#mindcv.data.create_transforms) parameters:

- dataset_name: if ‘ ’, customized dataset. Currently, apply the same transform pipeline as ImageNet. if standard dataset name is given including imagenet, cifar10, mnist, preset transforms will be returned. Default: ‘ ’.

- image_resize: the image size after resize for adapting to network. Default: 224.

- is_training:  if True, augmentation will be applied if support. Default: False.

- **kwargs: additional args parsed to transforms_imagenet_train and transforms_imagenet_eval.

2. The [mindcv.data.create_loader](https://mindcv.readthedocs.io/en/latest/api/mindcv.data.html#mindcv.data.create_loader) function is used for data conversion and batch split loading. We need to pass in the transform_list returned by [create_transforms](https://mindcv.readthedocs.io/en/latest/api/mindcv.data.html#mindcv.data.create_transforms).



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

[create_loader](https://mindcv.readthedocs.io/en/latest/api/mindcv.data.html#mindcv.data.create_loader) parameters:

- dataset: dataset object created by create_dataset.

- batch_size: The number of rows each batch is created with. An int or callable object which takes exactly 1 parameter, BatchInfo.

- drop_remainder: Determines whether to drop the last block whose data row number is less than batch size (default=False). If True, and if there are less than batch_size rows available to make the last batch, then those rows will be dropped and not propagated to the child node.

- is_training: whether it is in train mode. Default: False.

- mixup: mixup alpha, mixup will be enbled if > 0. (default=0.0).

- cutmix: cutmix alpha, cutmix will be enabled if > 0. (default=0.0). This operation is experimental.

- cutmix_prob: prob of doing cutmix for an image (default=0.0)

- num_classes: the number of classes. Default: 1000.

- transform: the list of transformations that wil be applied on the image, which is obtained by create_transform. If None, the default imagenet transformation for evaluation will be applied. Default: None.

- target_transform: the list of transformations that will be applied on the label. If None, the label will be converted to the type of ms.int32. Default: None.

- num_parallel_workers: Number of workers(threads) to process the dataset in parallel (default=None).

- python_multiprocessing: Parallelize Python operations with multiple worker processes. This option could be beneficial if the Python operation is computational heavy (default=False).


>Avoid repeatedly executing a single cell of [create_loader](https://mindcv.readthedocs.io/en/latest/api/mindcv.data.html#mindcv.data.create_loader) in notebook, or execute again after executing [create_dataset](https://mindcv.readthedocs.io/en/latest/api/mindcv.data.html#mindcv.data.create_dataset).

## Model Creation and Loading

Use the [create_model](https://mindcv.readthedocs.io/en/latest/api/mindcv.models.html#mindcv.models.create_model) interface to obtain the instantiated DenseNet and load the pretraining weight densenet_121_224.ckpt (obtained from ImageNet dataset training).


```python
from mindcv.models import create_model

# nstantiate the DenseNet121 model and load the pretraining weights.
network = create_model(model_name='densenet121', num_classes=num_classes, pretrained=True)
```

    32293888B [00:01, 28754770.92B/s]
    [WARNING] ME(1769:281472959711936,MainProcess):2022-12-21-16:03:22.690.392 [mindspore/train/serialization.py:712] For 'load_param_into_net', 2 parameters in the 'net' are not loaded, because they are not in the 'parameter_dict', please check whether the network structure is consistent when training and loading checkpoint.
    [WARNING] ME(1769:281472959711936,MainProcess):2022-12-21-16:03:22.691.960 [mindspore/train/serialization.py:714] classifier.weight is not loaded.
    [WARNING] ME(1769:281472959711936,MainProcess):2022-12-21-16:03:22.692.908 [mindspore/train/serialization.py:714] classifier.bias is not loaded.


> Because the number of classes required by Cifar10 and ImageNet datasets is different, the classifier parameters cannot be shared, and the warning that the classifier parameters cannot be loaded does not affect the fine tuning.

[create_model](https://mindcv.readthedocs.io/en/latest/api/mindcv.models.html#mindcv.models.create_model) parameters:

- model_name: The name of model.

- num_classes: The number of classes. Default: 1000.

- pretrained: Whether to load the pretrained model. Default: False.

- in_channels: The input channels. Default: 3.

- checkpoint_path: The path of checkpoint files. Default: “”.

- use_ema: Whether use ema method. Default: False.

Use the [mindcv.loss.create_loss](https://mindcv.readthedocs.io/en/latest/api/mindcv.loss.html#mindcv.loss.create_loss) interface to create a loss function (cross_entropy loss).

## Model Training

By [create_loss](https://mindcv.readthedocs.io/en/latest/api/mindcv.loss.html#mindcv.loss.create_loss) interface obtains loss function.


```python
from mindcv.loss import create_loss

loss = create_loss(name='CE')
```

[create_loss](https://mindcv.readthedocs.io/en/latest/api/mindcv.loss.html#mindcv.loss.create_loss) parameters:

- name: loss name, ‘CE’ for cross_entropy. ‘BCE’: binary cross entropy. Default: ‘CE’.

- weight: Class weight. A rescaling weight given to the loss of each batch element. If given, has to be a Tensor of size ‘nbatch’. Data type must be float16 or float32.

- reduction: Apply specific reduction method to the output: ‘mean’ or ‘sum’. By default, the sum of the output will be divided by the number of elements in the output. ‘sum’: the output will be summed. Default:’mean’.

- label_smoothing: Label smoothing factor, a regularization tool used to prevent the model from overfitting when calculating Loss. The value range is [0.0, 1.0]. Default: 0.0.

- aux_factor: Auxiliary loss factor. Set aux_fuactor > 0.0 if the model has auxilary logit outputs (i.e., deep supervision), like inception_v3. Default: 0.0.

Use [create_scheduler](https://mindcv.readthedocs.io/en/latest/api/mindcv.scheduler.html#mindcv.scheduler.create_scheduler) interface sets the learning rate scheduler.


```python
from mindcv.scheduler import create_scheduler

# learning rate scheduler
lr_scheduler = create_scheduler(steps_per_epoch=num_batches,
                                scheduler='constant',
                                lr=0.0001)
```

[create_scheduler](https://mindcv.readthedocs.io/en/latest/api/mindcv.scheduler.html#mindcv.scheduler.create_scheduler) parameters:

- steps_pre_epoch: number of steps per epoch.

- scheduler: scheduler name like ‘constant’, ‘cosine_decay’, ‘step_decay’, ‘exponential_decay’, ‘polynomial_decay’, ‘multi_step_decay’. Default: ‘constant’.

- lr: learning rate value. Default: 0.01.

- min_lr: lower lr bound for ‘cosine_decay’ schedulers. Default: 1e-6.

- warmup_epochs: epochs to warmup LR, if scheduler supports. Default: 3.

- warmup_factor: the warmup phase of scheduler is a linearly increasing lr, the beginning factor is warmup_factor, i.e., the lr of the first step/epoch is lr*warmup_factor, and the ending lr in the warmup phase is lr. Default: 0.0.

- decay_epochs: for ‘cosine_decay’ schedulers, decay LR to min_lr in decay_epochs. For ‘step_decay’ scheduler, decay LR by a factor of decay_rate every decay_epochs. Default: 10.

- decay_rate: LR decay rate (default: 0.9).

- milestones: list of epoch milestones for ‘multi_step_decay’ scheduler. Must be increasing.

- num_epochs: number of total epochs.

- lr_epoch_stair: If True, LR will be updated in the beginning of each new epoch and the LR will be consistent for each batch in one epoch. Otherwise, learning rate will be updated dynamically in each step. (default=False).


Use [create_optimizer](https://mindcv.readthedocs.io/en/latest/api/mindcv.optim.html#mindcv.optim.create_optimizer) interface creates an optimizer.


```python
from mindcv.optim import create_optimizer

# create optimizer
opt = create_optimizer(network.trainable_params(), opt='adam', lr=lr_scheduler)
```

[create_optimizer](https://mindcv.readthedocs.io/en/latest/api/mindcv.optim.html#mindcv.optim.create_optimizer) parameters:

- params: network parameters. Union[list[Parameter],list[dict]], which must be the list of parameters or list of dicts. When the list element is a dictionary, the key of the dictionary can be “params”, “lr”, “weight_decay”,”grad_centralization” and “order_params”.

- opt: Wrapped optimizer. You could choose like ‘sgd’, ‘nesterov’, ‘momentum’, ‘adam’, ‘adamw’, ‘rmsprop’, ‘adagrad’, ‘lamb’. ‘adam’ is the default choise for convolution-based networks. ‘adamw’ is recommended for ViT-based networks. Default: ‘adam’.

- lr: learning rate, float or lr scheduler. Fixed and dynamic learning rate are supported. Default: 1e-3.

- weight_decay: weight decay factor. It should be noted that weight decay can be a constant value or a Cell. It is a Cell only when dynamic weight decay is applied. Dynamic weight decay is similar to dynamic learning rate, users need to customize a weight decay schedule only with global step as input, and during training, the optimizer calls the instance of WeightDecaySchedule to get the weight decay value of current step. Default: 0.

- momentum: momentum if the optimizer supports. Default: 0.9.

- nesterov: whether to use Nesterov Accelerated Gradient (NAG) algorithm to update the gradients. Default: False.

- filter_bias_and_bn: whether to filter batch norm parameters and bias from weight decay. If True, weight decay will not apply on BN parameters and bias in Conv or Dense layers. Default: True.

- loss_scale: A floating point value for the loss scale, which must be larger than 0.0. Default: 1.0.

- checkpoint_path: optimizer checkpoint path.

- eps: Term Added to the Denominator to Improve Numerical Stability. default: 1e-10.

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



```python
# Load validation dataset
dataset_val = create_dataset(name='cifar10', root=cifar10_dir, split='test', shuffle=True, num_parallel_workers=num_workers, download=download)

# Perform data enhancement operations to generate the required dataset.
loader_val = create_loader(dataset=dataset_val,
                           batch_size=64,
                           is_training=False,
                           num_classes=num_classes,
                           transform=trans,
                           num_parallel_workers=num_workers)
```

Load the fine-tuning parameter file (densenet121-cifar10-5_782.ckpt) to the model.

Encapsulate inferable instances according to the parameters passed in by the user, load the validation data set, and verify the precision of the fine tuned DenseNet121 model.


```python
# Verify the accuracy of DenseNet121 after fine-tune
acc = model.eval(loader_val, dataset_sink_mode=False)
print(acc)
```

    [WARNING] DEVICE(1769,ffff87c70ac0,python):2022-12-21-16:24:11.927.472 [mindspore/ccsrc/plugin/device/ascend/hal/device/kernel_select_ascend.cc:330] FilterRaisedOrReducePrecisionMatchedKernelInfo] Operator:[Default/network-WithLossCell/_loss_fn-CrossEntropySmooth/GatherD-op24314] don't support int64, reduce precision from int64 to int32.


    {'accuracy': 0.951}


    [WARNING] DEVICE(1769,ffff87c70ac0,python):2022-12-21-16:25:01.871.273 [mindspore/ccsrc/plugin/device/ascend/hal/device/kernel_select_ascend.cc:330] FilterRaisedOrReducePrecisionMatchedKernelInfo] Operator:[Default/network-WithLossCell/_loss_fn-CrossEntropySmooth/GatherD-op27139] don't support int64, reduce precision from int64 to int32.


## Use YAML files for model training and validation

We can also use the yaml file with the model parameters set directly to quickly train and verify the model through `train.py` and `validate.py` scripts. The following is an example of training SqueezenetV1 on [ImageNet](https://www.image-net.org/challenges/LSVRC/2012/index.php) (you need to download ImageNet to the directory in advance)


> For detailed tutorials, please refer to the [tutorial](https://mindcv.readthedocs.io/en/latest/tutorials/learn_about_config.html).




```python
!git clone https://github.com/mindspore-lab/mindcv.git
!cd mindcv
```


```python
# standalone training on a CPU/GPU/Ascend device
!python train.py -c configs/squeezenet/squeezenet_1.0_gpu.yaml --data_dir /path/to/dataset --distribute False
```


```python
!python validate.py -c configs/squeezenet/squeezenet_1.0_gpu.yaml --data_dir /path/to/dataset --ckpt_path /path/to/ckpt
```
