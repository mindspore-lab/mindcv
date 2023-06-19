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

<div align="center">

|     Model      | Context  | Top-1 (%) | Top-5 (%) | Params (M) | Recipe                                                                                                  | Download                                                                                         |
|:--------------:|:--------:|:---------:|:---------:|:----------:|---------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| poolformer_s12 | D910x8-G |   77.33   |   93.34   |   11.92    | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/poolformer/poolformer_s12_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/poolformer/poolformer_s12-5be5c4e4.ckpt) |

</div>

#### Notes

- Context: Training context denoted as {device}x{pieces}-{MS mode}, where mindspore mode can be G - graph mode or F - pynative mode with ms function. For example, D910x8-G is for training on 8 pieces of Ascend 910 NPU using graph mode.
- Top-1 and Top-5: Accuracy reported on the validation set of ImageNet-1K.

## Quick Start

### Preparation

#### Installation

Please refer to the [installation instruction](https://github.com/mindspore-lab/mindcv#installation) in MindCV.

#### Dataset Preparation

Please download the [ImageNet-1K](https://www.image-net.org/challenges/LSVRC/2012/index.php) dataset for model training and validation.

### Training

- Distributed Training

It is easy to reproduce the reported results with the pre-defined training recipe. For distributed training on multiple Ascend 910 devices, please run

```shell
# distributed training on multiple GPU/Ascend devices
mpirun -n 8 python train.py --config configs/poolformer/poolformer_s12_ascend.yaml --data_dir /path/to/imagenet
```
> If the script is executed by the root user, the `--allow-run-as-root` parameter must be added to `mpirun`.

Similarly, you can train the model on multiple GPU devices with the above `mpirun` command.

For detailed illustration of all hyper-parameters, please refer to [config.py](https://github.com/mindspore-lab/mindcv/blob/main/config.py).

**Note:** As the global batch size (batch_size x num_devices) is an important hyper-parameter, it is recommended to keep the global batch size unchanged for reproduction or adjust the learning rate linearly to a new global batch size.

* Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on a CPU/GPU/Ascend device
python train.py --config configs/poolformer/poolformer_s12_ascend.yaml --data_dir /path/to/imagenet --distribute False
```

### Validation

```
validation of poolformer has to be done in amp O3 mode which is not supported, coming soon...
```

### Deployment

To deploy online inference services with the trained model efficiently, please refer to the [deployment tutorial](https://mindspore-lab.github.io/mindcv/tutorials/deployment/).

## References

[1]. Yu W, Luo M, Zhou P, et al. Metaformer is actually what you need for vision[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022: 10819-10829.
