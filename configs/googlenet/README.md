# GoogLeNet
> [GoogLeNet: Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842)

## Introduction

GoogLeNet is a new deep learning structure proposed by Christian Szegedy in 2014. Prior to this, AlexNet, VGG and other
structures achieved better training effects by increasing the depth (number of layers) of the network, but the increase
in the number of layers It will bring many negative effects, such as overfit, gradient disappearance, gradient
explosion, etc. The proposal of inception improves the training results from another perspective: it can use computing
resources more efficiently, and can extract more features under the same amount of computing, thereby improving the
training results.[[1](#references)]

<p align="center">
  <img src="https://user-images.githubusercontent.com/53842165/210749903-5ff23c0e-547f-487d-bb64-70b6e99031ea.jpg" width=180 />
</p>
<p align="center">
  <em>Figure 1. Architecture of GoogLeNet [<a href="#references">1</a>] </em>
</p>

## Results

Our reproduced model performance on ImageNet-1K is reported as follows.

- ascend 910* with graph mode


<div align="center">


| model     | top-1 (%) | top-5 (%) | params (M) | batch size | cards | ms/step | jit_level | recipe                                                                                            | download                                                                                                 |
| --------- | --------- | --------- | ---------- | ---------- | ----- | ------- | --------- | ------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| googlenet | 72.89     | 90.89     | 6.99       | 32         | 8     | 23.5    | O2        | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/googlenet/googlenet_ascend.yaml) | [weights](https://download-mindspore.osinfra.cn/toolkits/mindcv/googlenet/googlenet-de74c31d-910v2.ckpt) |

</div>

- ascend 910 with graph mode

<div align="center">


| model     | top-1 (%) | top-5 (%) | params (M) | batch size | cards | ms/step | jit_level | recipe                                                                                            | download                                                                                   |
| --------- | --------- | --------- | ---------- | ---------- | ----- | ------- | --------- | ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| googlenet | 72.68     | 90.89     | 6.99       | 32         | 8     | 21.40   | O2        | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/googlenet/googlenet_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/googlenet/googlenet-5552fcd3.ckpt) |

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

* Distributed Training

It is easy to reproduce the reported results with the pre-defined training recipe. For distributed training on multiple Ascend 910 devices, please run

```shell
# distributed training on multiple NPU devices
msrun --bind_core=True --worker_num 8 python train.py --config configs/googlenet/googlenet_ascend.yaml --data_dir /path/to/imagenet
```




For detailed illustration of all hyper-parameters, please refer to [config.py](https://github.com/mindspore-lab/mindcv/blob/main/config.py).

**Note:**  As the global batch size  (batch_size x num_devices) is an important hyper-parameter, it is recommended to keep the global batch size unchanged for reproduction or adjust the learning rate linearly to a new global batch size.

* Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on single NPU device
python train.py --config configs/googlenet/googlenet_ascend.yaml --data_dir /path/to/dataset --distribute False
```

### Validation

To validate the accuracy of the trained model, you can use `validate.py` and parse the checkpoint path with `--ckpt_path`.

```shell
python validate.py -c configs/googlenet/googlenet_ascend.yaml --data_dir /path/to/imagenet --ckpt_path /path/to/ckpt
```

## References

[1] Szegedy C, Liu W, Jia Y, et al. Going deeper with convolutions[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2015: 1-9.
