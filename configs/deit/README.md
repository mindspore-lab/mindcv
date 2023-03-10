
# DeiT
> [Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877)

## Introduction

DeiT: Data-efficient Image Transformers

## Results

**Implementation and configs for training were taken and adjusted from [this repository](https://gitee.com/cvisionlab/models/tree/deit/release/research/cv/DeiT), which implements Twins models in mindspore.**

Our reproduced model performance on ImageNet-1K is reported as follows.

<div align="center">

| Model    | Context  | Top-1 (%) | Top-5 (%) | Params (M) | Recipe                                                                                        | Download                                                                               |
|----------|----------|-----------|-----------|------------|-----------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------|
| deit_base | Converted from PyTorch | 81.62     | 95.58     | -  | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/deit/deit_b.yaml) | [weights](https://storage.googleapis.com/huawei-mindspore-hk/DeiT/Converted/deit_base_patch16_224.ckpt) |
| deit_base | 8xRTX3090 | 72.29 | 89.93 | - | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/deit/deit_b.yaml) | [weights](https://storage.googleapis.com/huawei-mindspore-hk/DeiT/deit_base_patch16_224_acc%3D0.725.ckpt)
| deit_small | Converted from PyTorch | 79.39 | 94.80 | - | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/deit/deit_b.yaml) | [weights](https://storage.googleapis.com/huawei-mindspore-hk/DeiT/Converted/deit_small_patch16_224.ckpt) |
| deit_tiny | Converted from PyTorch | 71.58 | 90.76 | - | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/deit/deit_b.yaml) | [weights](https://storage.googleapis.com/huawei-mindspore-hk/DeiT/Converted/deit_tiny_patch16_224.ckpt) |

</div>

#### Notes

- Context: The weights in the table were taken from [official repository](https://github.com/facebookresearch/deit) and converted to mindspore format
- Top-1 and Top-5: Accuracy reported on the validation set of ImageNet-1K.

## Quick Start

### Preparation

#### Installation
Please refer to the [installation instruction](https://github.com/mindspore-ecosystem/mindcv#installation) in MindCV.

#### Dataset Preparation
Please download the [ImageNet-1K](https://www.image-net.org/challenges/LSVRC/2012/index.php) dataset for model training and validation.

### Training

* Distributed Training


```shell
# distrubted training on multiple GPU/Ascend devices
mpirun -n 8 python train.py --config configs/deit/deit_b.yaml --data_dir /path/to/imagenet --distributed True
```

> If the script is executed by the root user, the `--allow-run-as-root` parameter must be added to `mpirun`.

Similarly, you can train the model on multiple GPU devices with the above `mpirun` command.

For detailed illustration of all hyper-parameters, please refer to [config.py](https://github.com/mindspore-lab/mindcv/blob/main/config.py).

**Note:**  As the global batch size  (batch_size x num_devices) is an important hyper-parameter, it is recommended to keep the global batch size unchanged for reproduction or adjust the learning rate linearly to a new global batch size.

* Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on a CPU/GPU/Ascend device
python train.py --config configs/deit/deit_b.yaml --data_dir /path/to/dataset --distribute False
```

### Validation

To validate the accuracy of the trained model, you can use `validate.py` and parse the checkpoint path with `--ckpt_path`.

```shell
python validate.py -c configs/deit/deit_b.yaml --data_dir /path/to/imagenet --ckpt_path /path/to/ckpt
```

### Deployment

Please refer to the [deployment tutorial](https://github.com/mindspore-lab/mindcv/blob/main/tutorials/deployment.md) in MindCV.

## References

Paper - https://arxiv.org/pdf/2012.12877.pdf

Official repo - https://github.com/facebookresearch/deit

Mindspore implementation - https://gitee.com/cvisionlab/models/tree/deit/release/research/cv/DeiT
