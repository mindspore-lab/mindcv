# RepMLPNet

> [RepMLPNet: Hierarchical Vision MLP with Re-parameterized Locality](https://arxiv.org/abs/2112.11081)

## Introduction

Compared to convolutional layers, fully-connected (FC) layers are better at modeling the long-range dependencies
but worse at capturing the local patterns, hence usually less favored for image recognition. In this paper, the authors
propose a
methodology, Locality Injection, to incorporate local priors into an FC layer via merging the trained parameters of a
parallel conv kernel into the FC kernel. Locality Injection can be viewed as a novel Structural Re-parameterization
method since it equivalently converts the structures via transforming the parameters. Based on that, the authors propose
a
multi-layer-perceptron (MLP) block named RepMLP Block, which uses three FC layers to extract features, and a novel
architecture named RepMLPNet. The hierarchical design distinguishes RepMLPNet from the other concurrently proposed
vision MLPs.
As it produces feature maps of different levels, it qualifies as a backbone model for downstream tasks like semantic
segmentation.
Their results reveal that 1) Locality Injection is a general methodology for MLP models; 2) RepMLPNet has favorable
accuracy-efficiency
trade-off compared to the other MLPs; 3) RepMLPNet is the first MLP that seamlessly transfer to Cityscapes semantic
segmentation.

![RepMLP](https://user-images.githubusercontent.com/74176172/210046952-c4f05321-76e9-4d7a-b419-df91aac64cdf.png)
Figure 1. RepMLP Block.[[1](#References)]

## Requirements
| mindspore | ascend driver |  firmware   | cann toolkit/kernel |
| :-------: | :-----------: | :---------: | :-----------------: |
|   2.5.0   |   24.1.0      | 7.5.0.3.220 |     8.0.0.beta1     |


## Quick Start

### Preparation

#### Installation

Please refer to the [installation instruction](https://mindspore-lab.github.io/mindcv/installation/) in MindCV.

#### Dataset Preparation

Please download the [ImageNet-1K](https://www.image-net.org/challenges/LSVRC/2012/index.php) dataset for model training
and validation.

### Training

- Distributed Training

It is easy to reproduce the reported results with the pre-defined training recipe. For distributed training on multiple
Ascend 910 devices, please run

```shell
# distributed training on multiple NPU devices
msrun --bind_core=True --worker_num 8 python train.py --config configs/repmlp/repmlp_t224_ascend.yaml --data_dir /path/to/imagenet
```



For detailed illustration of all hyper-parameters, please refer
to [config.py](https://github.com/mindspore-lab/mindcv/blob/main/config.py).

**Note:** As the global batch size (batch_size x num_devices) is an important hyper-parameter, it is recommended to keep
the global batch size unchanged for reproduction or adjust the learning rate linearly to a new global batch size.

* Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on single NPU device
python train.py --config configs/repmlp/repmlp_t224_ascend.yaml --data_dir /path/to/imagenet --distribute False
```

### Validation

To validate the accuracy of the trained model, you can use `validate.py` and parse the checkpoint path
with `--ckpt_path`.

```shell
python validate.py --model=repmlp_t224 --data_dir /path/to/imagenet --ckpt_path /path/to/ckpt
```

## Performance

Our reproduced model performance on ImageNet-1K is reported as follows.

Experiments are tested on ascend 910* with mindspore 2.5.0 graph mode.

*coming soon*

### Notes

- top-1 and top-5: Accuracy reported on the validation set of ImageNet-1K.

## References

[1]. Ding X, Chen H, Zhang X, et al. Repmlpnet: Hierarchical vision mlp with re-parameterized locality[C]//Proceedings
of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022: 578-587.
