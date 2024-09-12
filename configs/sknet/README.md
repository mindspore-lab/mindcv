# SKNet

> [Selective Kernel Networks](https://arxiv.org/pdf/1903.06586)

## Introduction

The local receptive fields (RFs) of neurons in the primary visual cortex (V1) of cats [[1](#references)] have inspired
the
construction of Convolutional Neural Networks (CNNs) [[2](#references)] in the last century, and it continues to inspire
mordern CNN
structure construction. For instance, it is well-known that in the visual cortex, the RF sizes of neurons in the
same area (e.g.,V1 region) are different, which enables the neurons to collect multi-scale spatial information in the
same processing stage. This mechanism has been widely adopted in recent Convolutional Neural Networks (CNNs).
A typical example is InceptionNets [[3](#references), [4](#references), [5](#references), [6](#references)], in which a
simple concatenation is designed to aggregate
multi-scale information from, e.g., 3×3, 5×5, 7×7 convolutional kernels inside the “inception” building block.

<p align="center">
  <img src="https://user-images.githubusercontent.com/22534574/225858259-405e225a-a5d9-4db9-a823-703c89381a2f.png" width=800 />
</p>
<p align="center">
  <em>Figure 1. Selective Kernel Convolution.</em>
</p>

## Results

Our reproduced model performance on ImageNet-1K is reported as follows.

performance tested on ascend 910*(8p) with graph mode

<div align="center">

|   Model    | Top-1 (%) | Top-5 (%) | ms/step | Params (M) | Batch Size | Recipe                                                                                         | Download                                                                                              |
| :--------: | :-------: | :-------: | :-----: | :--------: | ---------- | ---------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| skresnet18 |   72.85   |   90.83   |  48.35  |   11.97    | 64         | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/sknet/skresnet18_ascend.yaml) | [weights](https://download-mindspore.osinfra.cn/toolkits/mindcv/sknet/skresnet18-9d8b1afc-910v2.ckpt) |

</div>

performance tested on ascend 910(8p) with graph mode

<div align="center">

|   Model    | Top-1 (%) | Top-5 (%) | Params (M) | Batch Size | Recipe                                                                                         | Download                                                                                |
|:----------:|:---------:|:---------:|:----------:|------------|------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|
| skresnet18 |   73.09   |   91.20   |   11.97    | 64         | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/sknet/skresnet18_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/sknet/skresnet18-868228e5.ckpt) |

</div>

#### Notes

- Top-1 and Top-5: Accuracy reported on the validation set of ImageNet-1K.

## Quick Start

### Preparation

#### Installation

Please refer to the [installation instruction](https://github.com/mindspore-lab/mindcv#installation) in MindCV.

#### Dataset Preparation

Please download the [ImageNet-1K](https://www.image-net.org/challenges/LSVRC/2012/index.php) dataset for model training
and validation.

### Training

<!--- Guideline: Avoid using shell script in the command line. Python script preferred. -->

* Distributed Training

It is easy to reproduce the reported results with the pre-defined training recipe. For distributed training on multiple
Ascend 910 devices, please run

```shell
# distributed training on multiple GPU/Ascend devices
msrun --bind_core=True --worker_num 8 python train.py --config configs/sknet/skresnext50_32x4d_ascend.yaml --data_dir /path/to/imagenet
```

Similarly, you can train the model on multiple GPU devices with the above `msrun` command.

For detailed illustration of all hyper-parameters, please refer
to [config.py](https://github.com/mindspore-lab/mindcv/blob/main/config.py).

**Note:**  As the global batch size  (batch_size x num_devices) is an important hyper-parameter, it is recommended to
keep the global batch size unchanged for reproduction or adjust the learning rate linearly to a new global batch size.

* Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on a CPU/GPU/Ascend device
python train.py --config configs/sknet/skresnext50_32x4d_ascend.yaml --data_dir /path/to/dataset --distribute False
```

### Validation

To validate the accuracy of the trained model, you can use `validate.py` and parse the checkpoint path
with `--ckpt_path`.

```
python validate.py -c configs/sknet/skresnext50_32x4d_ascend.yaml --data_dir /path/to/imagenet --ckpt_path /path/to/ckpt
```

### Deployment

To deploy online inference services with the trained model efficiently, please refer to
the [deployment tutorial](https://mindspore-lab.github.io/mindcv/tutorials/deployment/).

## References

<!--- Guideline: Citation format GB/T 7714 is suggested. -->
[1] D. H. Hubel and T. N. Wiesel. Receptive fields, binocular interaction and functional architecture in the cat’s
visual
cortex. The Journal of Physiology, 1962.

[2] Y . LeCun, B. Boser, J. S. Denker, D. Henderson, R. E. Howard, W. Hubbard, and L. D. Jackel. Backpropagation
applied to handwritten zip code recognition. Neural Computation, 1989.

[3] C. Szegedy, V . V anhoucke, S. Ioffe, J. Shlens, and Z. Wojna. Rethinking the inception architecture for computer
vision. In
CVPR, 2016.

[4] S. Ioffe and C. Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate
shift.
arXiv preprint arXiv:1502.03167, 2015.

[5] C. Szegedy, V . V anhoucke, S. Ioffe, J. Shlens, and Z. Wojna. Rethinking the inception architecture for computer
vision. In
CVPR, 2016.

[6] C. Szegedy, S. Ioffe, V . V anhoucke, and A. A. Alemi. Inception-v4, inception-resnet and the impact of residual
connections on learning. In AAAI, 2017.
