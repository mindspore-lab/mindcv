
# Twins
> [Twins: Revisiting the Design of Spatial Attention in Vision Transformers](https://openreview.net/pdf?id=5kTlVBkzSRx)

## Introduction

Very recently, a variety of vision transformer architectures for dense prediction tasks have been proposed and they show that the design of spatial attention is critical to their success in these tasks. In this work, we revisit the design of the spatial attention and demonstrate that a carefully-devised yet simple spatial attention mechanism performs favourably against the state-of-the-art schemes. As a result, we propose two vision transformer architectures, namely, Twins- PCPVT and Twins-SVT. Our proposed architectures are highly-efficient and easy to implement, only involving matrix multiplications that are highly optimized in modern deep learning frameworks. More importantly, the proposed architectures achieve excellent performance on a wide range of visual tasks including image- level classification as well as dense detection and segmentation. The simplicity and strong performance suggest that our proposed architectures may serve as stronger backbones for many vision tasks.

<img width="1285" alt="twins_svt_s" src="https://user-images.githubusercontent.com/41994229/224014703-ed5ee3ed-3e82-46fb-bd34-289519095a7e.png">

Twins-SVT-S Architecture (Right side shows the inside of two consecutive Transformer Encoders).

## Results

**Implementation and configs for training were taken and adjusted from [this repository](https://gitee.com/cvisionlab/models/tree/twins/release/research/cv/Twins), which implements Twins models in mindspore.**

Our reproduced model performance on ImageNet-1K is reported as follows.

<div align="center">

| Model    | Context  | Top-1 (%) | Top-5 (%) | Params (M) | Recipe                                                                                        | Download                                                                               |
|----------|----------|-----------|-----------|------------|-----------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------|
| svt_small | Converted from PyTorch | 81     | 95.38     | -       | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/twins/svt_s_gpu.yaml) | [weights](https://storage.googleapis.com/huawei-mindspore-hk/Twins/converted/svt_s_new.ckpt) |
| svt_base | Converted from PyTorch | 82.63 | 96.17 | - | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/twins/svt_s_gpu.yaml) | [weights](https://storage.googleapis.com/huawei-mindspore-hk/Twins/converted/svt_b_new.ckpt) |
| svt_large | Converted from PyTorch | 83.04 | 96.35 | - | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/twins/svt_s_gpu.yaml) | [weights](https://storage.googleapis.com/huawei-mindspore-hk/Twins/converted/svt_l_new.ckpt) |
| pcpvt_small | Converted from Pytorch | 80.58 | 95.40 | - |[yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/twins/pcpvt_l_gpu.yaml) | [weights](https://storage.googleapis.com/huawei-mindspore-hk/Twins/converted/pcpvt_s_new.ckpt) |
| pcpvt_base | Converted from Pytorch | 82.19 | 96.08 | - | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/twins/pcpvt_l_gpu.yaml) | [weights](https://storage.googleapis.com/huawei-mindspore-hk/Twins/converted/pcpvt_b_new.ckpt) |
| pcpvt_large | Converted from PyTorch | 82.51 | 96.37 | - | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/twins/pcpvt_l_gpu.yaml) | [weights](https://storage.googleapis.com/huawei-mindspore-hk/Twins/converted/pcpvt_l_new.ckpt)

</div>

#### Notes

- Context: The weights in the table were taken from [official repository](https://github.com/Meituan-AutoML/Twins) and converted to mindspore format
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
mpirun -n 8 python train.py --config configs/twins/svt_s_gpu.yaml --data_dir /path/to/imagenet --distributed True
```

> If the script is executed by the root user, the `--allow-run-as-root` parameter must be added to `mpirun`.

Similarly, you can train the model on multiple GPU devices with the above `mpirun` command.

For detailed illustration of all hyper-parameters, please refer to [config.py](https://github.com/mindspore-lab/mindcv/blob/main/config.py).

**Note:**  As the global batch size  (batch_size x num_devices) is an important hyper-parameter, it is recommended to keep the global batch size unchanged for reproduction or adjust the learning rate linearly to a new global batch size.

* Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on a CPU/GPU/Ascend device
python train.py --config configs/twins/svt__gpus.yaml --data_dir /path/to/dataset --distribute False
```

### Validation

To validate the accuracy of the trained model, you can use `validate.py` and parse the checkpoint path with `--ckpt_path`.

```shell
python validate.py -c configs/twins/svt_s_gpu.yaml --data_dir /path/to/imagenet --ckpt_path /path/to/ckpt
```

### Deployment

Please refer to the [deployment tutorial](https://github.com/mindspore-lab/mindcv/blob/main/tutorials/deployment.md) in MindCV.

## References

Paper - https://openreview.net/pdf?id=5kTlVBkzSRx

Official repo - https://github.com/Meituan-AutoML/Twins

Mindspore implementation - https://gitee.com/cvisionlab/models/tree/twins/release/research/cv/Twins
