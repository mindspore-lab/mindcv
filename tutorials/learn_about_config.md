# Understanding Model Configuration

[![下载Notebook](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.8/resource/_static/logo_notebook.png)](https://download.mindspore.cn/toolkits/mindcv/tutorials/learn_about_config.ipynb)&emsp;


`mindcv` can parse the yaml file of the model through the `argparse` library and `pyyaml` library to configure parameters. Let's use squeezenet_1.0 model as an example to explain how to configure the corresponding parameters.


## Basic Environment

1. Parameter description

- mode: Use graph mode (0) or pynative mode (1).

- distribute: Whether to use distributed.


2. Sample yaml file

```text
mode: 0
distribute: True
...
```

3. Parse parameter setting

```text
python train.py --mode 0 --distribute False ...
```

4. Corresponding code example

> `args.model` represents the parameter `mode`, `args.distribute` represents the parameter `distribute`。

```python
def train(args):
    ms.set_context(mode=args.mode)

    if args.distribute:
        init()
        device_num = get_group_size()
        rank_id = get_rank()
        ms.set_auto_parallel_context(device_num=device_num,
                                     parallel_mode='data_parallel',
                                     gradients_mean=True)
    else:
        device_num = None
        rank_id = None
    ...
```


## Dataset

1. Parameter description

- dataset: dataset name.

- data_dir: Path of dataset file.

- shuffle: whether to shuffle the dataset.

- dataset_download: whether to download the dataset.

- batch_size: The number of rows each batch.

- drop_remainder: Determines whether to drop the last block whose data row number is less than batch size.

- num_parallel_workers: Number of workers(threads) to process the dataset in parallel.


2. Sample yaml file

```text
dataset: 'imagenet'
data_dir: './imagenet2012'
shuffle: True
dataset_download: False
batch_size: 32
drop_remainder: True
num_parallel_workers: 8
...
```

3. Parse parameter setting

```text
python train.py ... --dataset imagenet --data_dir ./imagenet2012 --shuffle True \
            --dataset_download False --batch_size 32 --drop_remainder True \
            --num_parallel_workers 8 ...
```

4. Corresponding code example

```python
def train(args):
    ...
    dataset_train = create_dataset(
        name=args.dataset,
        root=args.data_dir,
        split='train',
        shuffle=args.shuffle,
        num_samples=args.num_samples,
        num_shards=device_num,
        shard_id=rank_id,
        num_parallel_workers=args.num_parallel_workers,
        download=args.dataset_download,
        num_aug_repeats=args.aug_repeats)

    ...
    target_transform = transforms.OneHot(num_classes) if args.loss == 'BCE' else None

    loader_train = create_loader(
        dataset=dataset_train,
        batch_size=args.batch_size,
        drop_remainder=args.drop_remainder,
        is_training=True,
        mixup=args.mixup,
        cutmix=args.cutmix,
        cutmix_prob=args.cutmix_prob,
        num_classes=args.num_classes,
        transform=transform_list,
        target_transform=target_transform,
        num_parallel_workers=args.num_parallel_workers,
    )

    ...
```

## Data Augmentation

1. Parameter description

- image_resize: the image size after resize for adapting to network.

- scale: random resize scale.

- ratio: random resize aspect ratio.

- hfilp: horizontal flip training aug probability.

- interpolation: image interpolation mode for resize operator.

- crop_pct: input image center crop percent.

- color_jitter: color jitter factor.

- re_prob: probability of performing erasing.

2. Sample yaml file

```text
image_resize: 224
scale: [0.08, 1.0]
ratio: [0.75, 1.333]
hflip: 0.5
interpolation: 'bilinear'
crop_pct: 0.875
color_jitter: [0.4, 0.4, 0.4]
re_prob: 0.5
...
```

3. Parse parameter setting

```text
python train.py ... --image_resize 224 --scale [0.08, 1.0] --ratio [0.75, 1.333] \
            --hflip 0.5 --interpolation "bilinear" --crop_pct 0.875 \
            --color_jitter [0.4, 0.4, 0.4] --re_prob 0.5 ...
```

4. Corresponding code example

```python
def train(args):
    ...
    transform_list = create_transforms(
        dataset_name=args.dataset,
        is_training=True,
        image_resize=args.image_resize,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        interpolation=args.interpolation,
        auto_augment=args.auto_augment,
        mean=args.mean,
        std=args.std,
        re_prob=args.re_prob,
        re_scale=args.re_scale,
        re_ratio=args.re_ratio,
        re_value=args.re_value,
        re_max_attempts=args.re_max_attempts
    )
    ...
```

## Model

1. Parameter description

- model: model name。

- num_classes: number of label classes.。

- pretrained: whether load pretrained model。

- ckpt_path: initialize model from this checkpoint.。

- keep_checkpoint_max: max number of checkpoint files。

- ckpt_save_dir: path of checkpoint.

- epoch_size: train epoch size.

- dataset_sink_mode: the dataset sink mode。

- amp_level: auto mixed precision level for saving memory and acceleration.

2. Sample yaml file

```text
model: 'squeezenet1_0'
num_classes: 1000
pretrained: False
ckpt_path: './squeezenet1_0_gpu.ckpt'
keep_checkpoint_max: 10
ckpt_save_dir: './ckpt/'
epoch_size: 200
dataset_sink_mode: True
amp_level: 'O0'
...
```

3. Parse parameter setting

```text
python train.py ... --model squeezenet1_0 --num_classes 1000 --pretrained False \
            --ckpt_path ./squeezenet1_0_gpu.ckpt --keep_checkpoint_max 10 \
            --ckpt_save_path ./ckpt/ --epoch_size 200 --dataset_sink_mode True \
            --amp_level O0 ...
```

4. Corresponding code example

```python
def train(args):
    ...
    network = create_model(model_name=args.model,
                    num_classes=args.num_classes,
                    in_channels=args.in_channels,
                    drop_rate=args.drop_rate,
                    drop_path_rate=args.drop_path_rate,
                    pretrained=args.pretrained,
                    checkpoint_path=args.ckpt_path,
                    ema=args.ema)
    ...
```

## Loss Function

1. Parameter description

- loss: name of loss function, BCE (BinaryCrossEntropy) or CE (CrossEntropy).

- label_smoothing: use label smoothing.

2. Sample yaml file

```text
loss: 'CE'
label_smoothing: 0.1
...
```

3. Parse parameter setting

```text
python train.py ... --loss CE --label_smoothing 0.1 ...
```

4. Corresponding code example

```python
def train(args):
    ...
    loss = create_loss(name=args.loss,
                 reduction=args.reduction,
                 label_smoothing=args.label_smoothing,
                 aux_factor=args.aux_factor)
    ...
```

## Learning Rate Scheduler

1. Parameter description

- scheduler: name of scheduler.

- min_lr: the minimum value of learning rate if scheduler supports.

- lr: learning rate.

- warmup_epochs: warmup epochs.

- decay_epochs: decay epochs.

2. Sample yaml file

```text
scheduler: 'cosine_decay'
min_lr: 0.0
lr: 0.01
warmup_epochs: 0
decay_epochs: 200
...
```

3. Parse parameter setting

```text
python train.py ... --scheduler cosine_decay --min_lr 0.0 --lr 0.01 \
             --warmup_epochs 0 --decay_epochs 200 ...
```

4. Corresponding code example

```python
def train(args):
    ...
    lr_scheduler = create_scheduler(num_batches,
                          scheduler=args.scheduler,
                          lr=args.lr,
                          min_lr=args.min_lr,
                          warmup_epochs=args.warmup_epochs,
                          warmup_factor=args.warmup_factor,
                          decay_epochs=args.decay_epochs,
                          decay_rate=args.decay_rate,
                          milestones=args.multi_step_decay_milestones,
                          num_epochs=args.epoch_size,
                          lr_epoch_stair=args.lr_epoch_stair)
    ...
```


## optimizer

1. Parameter description

- opt: name of optimizer。

- filter_bias_and_bn: filter Bias and BatchNorm.

- momentum: Hyperparameter of type float, means momentum for the moving average.

- weight_decay: weight decay（L2 penalty）。

- loss_scale: gradient scaling factor

- use_nesterov: whether enables the Nesterov momentum

2. Sample yaml file

```text
opt: 'momentum'
filter_bias_and_bn: True
momentum: 0.9
weight_decay: 0.00007
loss_scale: 1024
use_nesterov: False
...
```

3. Parse parameter setting

```text
python train.py ... --opt momentum --filter_bias_and_bn True --weight_decay 0.00007 \
              --loss_scale 1024 --use_nesterov False ...
```

4. Corresponding code example

```python
def train(args):
    ...
    if args.ema:
        optimizer = create_optimizer(network.trainable_params(),
                            opt=args.opt,
                            lr=lr_scheduler,
                            weight_decay=args.weight_decay,
                            momentum=args.momentum,
                            nesterov=args.use_nesterov,
                            filter_bias_and_bn=args.filter_bias_and_bn,
                            loss_scale=args.loss_scale,
                            checkpoint_path=opt_ckpt_path,
                            eps=args.eps)
     else:
        optimizer = create_optimizer(network.trainable_params(),
                            opt=args.opt,
                            lr=lr_scheduler,
                            weight_decay=args.weight_decay,
                            momentum=args.momentum,
                            nesterov=args.use_nesterov,
                            filter_bias_and_bn=args.filter_bias_and_bn,
                            checkpoint_path=opt_ckpt_path,
                            eps=args.eps)
    ...
```


## Combination of Yaml and Parse

You can override the parameter settings in the yaml file by using parse to set parameters. Take the following shell command as an example,

```shell
python train.py -c ./configs/squeezenet/squeezenet_1.0_gpu.yaml --data_dir ./data
```
The above command overwrites the value of `args.data_dir` parameter from ./imaget2012 in yaml file to ./data.
