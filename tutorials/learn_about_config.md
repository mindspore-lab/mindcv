# 了解模型配置

`mindcv`套件可以通过python的`argparse`库和`pyyaml`库解析模型的yaml文件来进行参数的配置，下面我们以squeezenet_1.0模型为例，解释如何配置相应的参数。


## 基础环境

1. 参数说明

- mode: 使用静态图模式（0）或动态图模式（1）。

- distribute: 是否使用分布式。


2. yaml文件样例

```text
mode: 0 
distribute: True 
...
```

3. parse参数设置

```text 
python train.py --mode 0 --distribute False ...
```

4. 对应的代码示例

> `args.model`代表参数`mode`, `args.distribute`代表参数`distribute`。

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


## 数据集

1. 参数说明

- dataset: 数据集名称。

- data_dir: 数据集文件所在路径。

- shuffle: 是否进行数据混洗。

- dataset_download: 是否下载数据集。

- batch_size: 每个批处理数据包含的数据条目。

- drop_remainder: 当最后一个批处理数据包含的数据条目小于 batch_size 时，是否将该批处理丢弃。

- num_parallel_workers: 读取数据的工作线程数。

2. yaml文件样例

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

3. parse参数设置

```text 
python train.py ... --dataset imagenet --data_dir ./imagenet2012 --shuffle True \
            --dataset_download False --batch_size 32 --drop_remainder True \
            --num_parallel_workers 8 ...
```

4. 对应的代码示例

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
        download=args.dataset_download)
    
    ...
    
    loader_train = create_loader(
        dataset=dataset_train,
        batch_size=args.batch_size,
        drop_remainder=args.drop_remainder,
        is_training=True,
        mixup=args.mixup,
        num_classes=args.num_classes,
        transform=transform_list,
        num_parallel_workers=args.num_parallel_workers,
    )
    
    ...
```

## 数据增强

1. 参数说明

- image_resize: 图像的输出尺寸大小。

- scale: 要裁剪的原始尺寸大小的各个尺寸的范围。

- ratio: 裁剪宽高比的范围。

- hfilp: 图像被翻转的概率。

- interpolation: 图像插值方式。

- crop_pct: 输入图像中心裁剪百分比。

- color_jitter: 颜色抖动因子（亮度调整因子，对比度调整因子，饱和度调整因子）。

- re_prob: 执行随机擦除的概率。

2. yaml文件样例

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

3. parse参数设置

```text
python train.py ... --image_resize 224 --scale [0.08, 1.0] --ratio [0.75, 1.333] \
            --hflip 0.5 --interpolation "bilinear" --crop_pct 0.875 \
            --color_jitter [0.4, 0.4, 0.4] --re_prob 0.5 ...
```

4. 对应的代码示例

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

## 模型

1. 参数说明

- model: 模型名称。

- num_classes: 分类的类别数。

- pretrained: 是否加载预训练模型。

- ckpt_path: 参数文件所在的路径。

- keep_checkpoint_max: 最多保存多少个checkpoint文件。

- ckpt_save_dir: 保存参数文件的路径。

- epoch_size: 训练执行轮次。

- dataset_sink_mode: 数据是否直接下沉至处理器进行处理。

- amp_level: 混合精度等级。

2. yaml文件样例

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

3. parse参数设置

```text
python train.py ... --model squeezenet1_0 --num_classes 1000 --pretrained False \
            --ckpt_path ./squeezenet1_0_gpu.ckpt --keep_checkpoint_max 10 \
            --ckpt_save_path ./ckpt/ --epoch_size 200 --dataset_sink_mode True \
            --amp_level O0 ...
```

4. 对应的代码示例

```python
def train(args):
    ...
    network = create_model(model_name=args.model,
                    num_classes=args.num_classes,
                    in_channels=args.in_channels,
                    drop_rate=args.drop_rate,
                    drop_path_rate=args.drop_path_rate,
                    pretrained=args.pretrained,
                    checkpoint_path=args.ckpt_path)
    ...
```

## 损失函数

1. 参数说明

- loss: 损失函数的简称。

- label_smoothing: 标签平滑值，用于计算Loss时防止模型过拟合的正则化手段。

2. yaml文件样例

```text
loss: 'CE'
label_smoothing: 0.1
...
```

3. parse参数设置

```text
python train.py ... --loss CE --label_smoothing 0.1 ...
```

4. 对应的代码示例

```python
def train(args):
    ...
    loss = create_loss(name=args.loss,
                 reduction=args.reduction,
                 label_smoothing=args.label_smoothing,
                 aux_factor=args.aux_factor)
    ...
```

## 学习率策略

1. 参数说明

- scheduler: 学习率策略的名称。

- min_lr: 学习率的最小值。

- lr: 学习率的最大值。

- warmup_epochs: 学习率warmup的轮次。

- decay_epochs: 进行衰减的step数。

2. yaml文件样例

```text
scheduler: 'warmup_cosine_decay'
min_lr: 0.0
lr: 0.01
warmup_epochs: 0
decay_epochs: 200
...
```

3. parse参数设置

```text
python train.py ... --scheduler warmup_cosine_decay --min_lr 0.0 --lr 0.01 \
             --warmup_epochs 0 --decay_epochs 200 ...
```

4. 对应的代码示例

```python
def train(args):
    ...
    lr_scheduler = create_scheduler(steps_per_epoch,
                          scheduler=args.scheduler,
                          lr=args.lr,
                          min_lr=args.min_lr,
                          warmup_epochs=args.warmup_epochs,
                          decay_epochs=args.decay_epochs,
                          decay_rate=args.decay_rate)
    ...
```


## 优化器

1. 参数说明

- opt: 优化器名称。

- filter_bias_and_bn: 参数中是否包含bias，gamma或者beta。

- momentum: 移动平均的动量。

- weight_decay: 权重衰减（L2 penalty）。

- loss_scale: 梯度缩放系数

- use_nesterov: 是否使用Nesterov Accelerated Gradient (NAG)算法更新梯度。

2. yaml文件样例

```text
opt: 'momentum'
filter_bias_and_bn: True
momentum: 0.9
weight_decay: 0.00007
loss_scale: 1024
use_nesterov: False
...
```

3. parse参数设置

```text
python train.py ... --opt momentum --filter_bias_and_bn True --weight_decay 0.00007 \
              --loss_scale 1024 --use_nesterov False ...
```

4. 对应的代码示例

```python
def train(args):
    ...
    optimizer = create_optimizer(network.trainable_params(),
                        opt=args.opt,
                        lr=lr_scheduler,
                        weight_decay=args.weight_decay,
                        momentum=args.momentum,
                        nesterov=args.use_nesterov,
                        filter_bias_and_bn=args.filter_bias_and_bn,
                        loss_scale=args.loss_scale)
    ...
```


## Yaml和Parse组合使用

使用parse设置参数可以覆盖yaml文件中的参数设置。以下面的shell命令为例，

```shell
python train.py -c ./configs/squeezenet/squeezenet_1.0_gpu.yaml --data_dir ./data
```
上面的命令将`args.data_dir`参数的值由yaml文件中的"./imagenet2012"覆盖为"./data"。



