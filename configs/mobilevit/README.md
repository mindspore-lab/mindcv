# MobileViT
> [MobileViT：Light-weight, General-purpose, and Mobile-friendly Vision Transformer](https://arxiv.org/pdf/2110.02178.pdf)


## Introduction

MobileViT, a light-weight and general-purpose vision transformer for mobile devices. MobileViT presents a different perspective for the global processing of information with transformers, i.e., transformers as convolutions. MobileViT significantly outperforms CNN- and ViT-based networks across different tasks and datasets. On the ImageNet-1k dataset, MobileViT achieves top-1 accuracy of 78.4% with about 6 million parameters, which is 3.2% and 6.2% more accurate than MobileNetv3 (CNN-based) and DeIT (ViT-based) for a similar number of parameters. On the MS-COCO object detection task, MobileViT is 5.7% more accurate than MobileNetv3 for a similar number of parameters.

<p align="center">
  <img src="https://user-images.githubusercontent.com/64628185/229476902-1b97496a-4a38-40ca-9e50-a88c52defcbb.png" width=800 />
</p>
<p align="center">
  <em>Figure 1. Architecture of MobileViT [<a href="#references">1</a>] </em>
</p>

## Requirements
| mindspore | ascend driver |  firmware   | cann toolkit/kernel |
| :-------: | :-----------: | :---------: | :-----------------: |
|   2.5.0   |   24.1.0      | 7.5.0.3.220 |     8.0.0.beta1     |



## Quick Start

### Preparation

#### Installation
Please refer to the [installation instruction](https://mindspore-lab.github.io/mindcv/installation/) in MindCV.

#### Dataset Preparation
Please download the [ImageNet-1K](https://www.image-net.org/challenges/LSVRC/2012/index.php) dataset for model training and validation.

### Training

* Distributed Training

It is easy to reproduce the reported results with the pre-defined training recipe. For distributed training on multiple Ascend 910 devices, please run

```shell
# distributed training on multiple NPU devices
msrun --bind_core=True --worker_num 8 python train.py --config configs/mobilevit/mobilevit_xx_small_ascend.yaml --data_dir /path/to/imagenet
```



For detailed illustration of all hyper-parameters, please refer to [config.py](https://github.com/mindspore-lab/mindcv/blob/main/config.py).

**Note:**  As the global batch size  (batch_size x num_devices) is an important hyper-parameter, it is recommended to keep the global batch size unchanged for reproduction or adjust the learning rate linearly to a new global batch size.

* Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on single NPU device
python train.py --config configs/mobilevit/mobilevit_xx_small_ascend.yaml --data_dir /path/to/dataset --distribute False
```

### Validation

To validate the accuracy of the trained model, you can use `validate.py` and parse the checkpoint path with `--ckpt_path`.

```
python validate.py -c configs/mobilevit/mobilevit_xx_small_ascend.yaml --data_dir /path/to/imagenet --ckpt_path /path/to/ckpt
```

## Performance

Our reproduced model performance on ImageNet-1K is reported as follows.

Experiments are tested on Ascend Atlas 800T A2 machines with mindspore 2.5.0 graph mode.


|  model name  |  params(M)   |  cards  |  batch size  |  resolution  |  jit level  |  graph compile  |  ms/step  |   img/s   |  acc@top1  |  acc@top5  |                                               recipe                                               |                                                  weight                                                   |
|:------------:|:------------:|:-------:|:------------:|:------------:|:-----------:|:---------------:|:---------:|:---------:|:----------:|:----------:|:--------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------:|
| mobilevit_xx_small | 1.27      | 8     | 64         | 256x256    | O2        | 437s          | 57.34   | 8929.19 | 67.11    | 87.85    | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/mobilevit/mobilevit_xx_small_ascend.yaml) | [weights](https://download-mindspore.osinfra.cn/toolkits/mindcv/mobilevit/mobilevit_xx_small-6f2745c3-910v2.ckpt) |


### Notes
- top-1 and top-5: Accuracy reported on the validation set of ImageNet-1K.
