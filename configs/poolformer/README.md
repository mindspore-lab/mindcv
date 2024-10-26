# PoolFormer

> [MetaFormer Is Actually What You Need for Vision](https://arxiv.org/abs/2111.11418)

## Introduction

Instead of designing complicated token mixer to achieve SOTA performance, the target of this work is to demonstrate the competence of Transformer models largely stem from the general architecture MetaFormer. Pooling/PoolFormer are just the tools to support the authors' claim.

![MetaFormer](https://user-images.githubusercontent.com/74176172/210046827-c218f5d3-1ee8-47bf-a78a-482d821ece89.png)
Figure 1. MetaFormer and performance of MetaFormer-based models on ImageNet-1K validation set. The authors argue that the competence of Transformer/MLP-like models primarily stem from the general architecture MetaFormer instead of the equipped specific token mixers. To demonstrate this, they exploit an embarrassingly simple non-parametric operator, pooling, to conduct extremely basic token mixing. Surprisingly, the resulted model PoolFormer consistently outperforms the DeiT and ResMLP as shown in (b), which well supports that MetaFormer is actually what we need to achieve competitive performance. RSB-ResNet in (b) means the results are from “ResNet Strikes Back” where ResNet is trained with improved training procedure for 300 epochs.

![PoolFormer](https://user-images.githubusercontent.com/74176172/210046845-6caa1574-b6a4-47f3-8298-c8ca3b4f8fa4.png)
Figure 2. (a) The overall framework of PoolFormer. (b) The architecture of PoolFormer block. Compared with Transformer block, it replaces attention with an extremely simple non-parametric operator, pooling, to conduct only basic token mixing.[[1](#References)]

## Results

Our reproduced model performance on ImageNet-1K is reported as follows.

- ascend 910* with graph mode

<div align="center">


|     model      | top-1 (%) | top-5 (%) | params (M) | batch size | cards | ms/step | jit_level | recipe                                                                                                  | download                                                                                                       |
| :------------: | :-------: | :-------: | :--------: | ---------- | ----- | ------- | --------- | ------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| poolformer_s12 |   77.49   |   93.55   |   11.92    | 128        | 8     | 211.81  | O2        | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/poolformer/poolformer_s12_ascend.yaml) | [weights](https://download-mindspore.osinfra.cn/toolkits/mindcv/poolformer/poolformer_s12-c7e14eea-910v2.ckpt) |

</div>

- ascend 910 with graph mode

<div align="center">


|     model      | top-1 (%) | top-5 (%) | params (M) | batch size | cards | ms/step | jit_level | recipe                                                                                                  | download                                                                                         |
| :------------: | :-------: | :-------: | :--------: | ---------- | ----- | ------- | --------- | ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| poolformer_s12 |   77.33   |   93.34   |   11.92    | 128        | 8     | 220.13  | O2        | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/poolformer/poolformer_s12_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/poolformer/poolformer_s12-5be5c4e4.ckpt) |

</div>

#### Notes
- Top-1 and Top-5: Accuracy reported on the validation set of ImageNet-1K.

## Quick Start

### Preparation

#### Installation

Please refer to the [installation instruction](https://mindspore-lab.github.io/mindcv/installation/) in MindCV.

#### Dataset Preparation

Please download the [ImageNet-1K](https://www.image-net.org/challenges/LSVRC/2012/index.php) dataset for model training and validation.

### Training

- Distributed Training

It is easy to reproduce the reported results with the pre-defined training recipe. For distributed training on multiple Ascend 910 devices, please run

```shell
# distributed training on multiple NPU devices
msrun --bind_core=True --worker_num 8 python train.py --config configs/poolformer/poolformer_s12_ascend.yaml --data_dir /path/to/imagenet
```



For detailed illustration of all hyper-parameters, please refer to [config.py](https://github.com/mindspore-lab/mindcv/blob/main/config.py).

**Note:** As the global batch size (batch_size x num_devices) is an important hyper-parameter, it is recommended to keep the global batch size unchanged for reproduction or adjust the learning rate linearly to a new global batch size.

* Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on single NPU device
python train.py --config configs/poolformer/poolformer_s12_ascend.yaml --data_dir /path/to/imagenet --distribute False
```

### Validation

```
validation of poolformer has to be done in amp O3 mode which is not supported, coming soon...
```

## References

[1]. Yu W, Luo M, Zhou P, et al. Metaformer is actually what you need for vision[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022: 10819-10829.
