# ResNet_EBV
> [Equiangular Basis Vectors](https://arxiv.org/abs/2303.11637)

## Introduction

EBVs provide a solution to the problem of classification with a large number of classes in resource-constrained environments. When the number of classes is C (e.g., C > 100,000), the number of trainable parameters in the final linear layer of a traditional ResNet-50 increases to 2048 * C. In contrast, EBVs reduce this by using fixed basis vectors for different classes, where the dimensionality is d (with d << C), and constraining the angles between these basis vectors during initialization. After that, EBVs are fixed, which reduces the number of trainable parameters to 2048 * d. EBVs can also be extended to other architectures.[[1](#references)]

<!-- <p align="center">
  <img src="https://user-images.githubusercontent.com/53842165/223672204-8ac59c6c-cd8a-45c2-945f-7e556c383056.jpg" width=500 />
</p>
<p align="center">
  <em>Figure 1. Comparisons between typical classification paradigms and Equiangular Basis Vectors (EBVs). [<a href="#references">1</a>] </em>
</p> -->

## Results

Our reproduced model performance on ImageNet-1K is reported as follows.

<div align="center">

| Model      | Context  | Top-1 (%) | Top-5 (%) | Params (M) | Recipe                                                                                           | Download                                                                                  |
|------------|----------|-----------|-----------|------------|--------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| resnet50_ebv  | D910x8-G | 78.12     | 93.80     | 27.55      | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/resnet_ebv/resnest50_ebv_ascend.yaml)  | \ |


</div>

#### Notes

- Context: Training context denoted as {device}x{pieces}-{MS mode}, where mindspore mode can be G - graph mode or F - pynative mode with ms function. For example, D910x8-G is for training on 8 pieces of Ascend 910 NPU using graph mode.
- Top-1 and Top-5: Accuracy reported on the validation set of ImageNet-1K.

## Quick Start

### Preparation

#### Installation
Please refer to the [installation instruction](https://github.com/mindspore-ecosystem/mindcv#installation) in MindCV.

#### Dataset Preparation
Please download the [ImageNet-1K](https://www.image-net.org/challenges/LSVRC/2012/index.php) dataset for model training and validation.

### Training

* Distributed Training

It is easy to reproduce the reported results with the pre-defined training recipe. For distributed training on multiple Ascend 910 devices, please run

```shell
# distributed training on multiple GPU/Ascend devices
mpirun -n 8 python train.py --config configs/resnest/resnest50_ebv_ascend.yaml --data_dir /path/to/imagenet
```

> If the script is executed by the root user, the `--allow-run-as-root` parameter must be added to `mpirun`.

Similarly, you can train the model on multiple GPU devices with the above `mpirun` command.

For detailed illustration of all hyper-parameters, please refer to [config.py](https://github.com/mindspore-lab/mindcv/blob/main/config.py).

**Note:**  As the global batch size  (batch_size x num_devices) is an important hyper-parameter, it is recommended to keep the global batch size unchanged for reproduction or adjust the learning rate linearly to a new global batch size.

* Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on a CPU/GPU/Ascend device
python train.py --config configs/resnest/resnest50_ebv_ascend.yaml --data_dir /path/to/dataset --distribute False
```

### Validation

To validate the accuracy of the trained model, you can use `validate.py` and parse the checkpoint path with `--ckpt_path`.

```shell
python validate.py -c configs/resnest/resnest50_ebv_ascend.yaml --data_dir /path/to/imagenet --ckpt_path /path/to/ckpt
```

### Deployment

Please refer to the [deployment tutorial](https://mindspore-lab.github.io/mindcv/tutorials/deployment/) in MindCV.

## References

[1] Shen Y, Sun X, Wei X S. Equiangular basis vectors[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023: 11755-11765.
