# MixNet
> [MixConv: Mixed Depthwise Convolutional Kernels](https://arxiv.org/abs/1907.09595)

## Introduction

Depthwise convolution is becoming increasingly popular in modern efficient ConvNets, but its kernel size is often
overlooked. In this paper, the authors systematically study the impact of different kernel sizes, and observe that
combining the benefits of multiple kernel sizes can lead to better accuracy and efficiency. Based on this observation,
the authors propose a new mixed depthwise convolution (MixConv), which naturally mixes up multiple kernel sizes in a
single convolution. As a simple drop-in replacement of vanilla depthwise convolution, our MixConv improves the accuracy
and efficiency for existing MobileNets on both ImageNet classification and COCO object detection.[[1](#references)]

<p align="center">
  <img src="https://user-images.githubusercontent.com/53842165/219263295-75de649e-d38b-4b05-bd26-1c96896f7e83.png" width=800 />
</p>
<p align="center">
  <em>Figure 1. Architecture of MixNet [<a href="#references">1</a>] </em>
</p>

## Results

Our reproduced model performance on ImageNet-1K is reported as follows.

- ascend 910* with graph mode

<div align="center">


| model    | top-1 (%) | top-5 (%) | params (M) | batch size | cards | ms/step | jit_level | recipe                                                                                        | download                                                                                             |
| -------- | --------- | --------- | ---------- | ---------- | ----- | ------- | --------- | --------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| mixnet_s | 75.58     | 95.54     | 4.17       | 128        | 8     | 228.03  | O2        | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/mixnet/mixnet_s_ascend.yaml) | [weights](https://download-mindspore.osinfra.cn/toolkits/mindcv/mixnet/mixnet_s-fe4fcc63-910v2.ckpt) |


</div>

- ascend 910 with graph mode

<div align="center">


| model    | top-1 (%) | top-5 (%) | params (M) | batch size | cards | ms/step | jit_level | recipe                                                                                        | download                                                                               |
| -------- | --------- | --------- | ---------- | ---------- | ----- | ------- | --------- | --------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| mixnet_s | 75.52     | 92.52     | 4.17       | 128        | 8     | 252.49  | O2        | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/mixnet/mixnet_s_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/mixnet/mixnet_s-2a5ef3a3.ckpt) |


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
# distrubted training on multiple GPU/Ascend devices
msrun --bind_core=True --worker_num 8 python train.py --config configs/mixnet/mixnet_s_ascend.yaml --data_dir /path/to/imagenet
```




For detailed illustration of all hyper-parameters, please refer to [config.py](https://github.com/mindspore-lab/mindcv/blob/main/config.py).

**Note:**  As the global batch size  (batch_size x num_devices) is an important hyper-parameter, it is recommended to keep the global batch size unchanged for reproduction or adjust the learning rate linearly to a new global batch size.

* Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on single NPU device
python train.py --config configs/mixnet/mixnet_s_ascend.yaml --data_dir /path/to/dataset --distribute False
```

### Validation

To validate the accuracy of the trained model, you can use `validate.py` and parse the checkpoint path with `--ckpt_path`.

```shell
python validate.py -c configs/mixnet/mixnet_s_ascend.yaml --data_dir /path/to/imagenet --ckpt_path /path/to/ckpt
```


## References

[1] Tan M, Le Q V. Mixconv: Mixed depthwise convolutional kernels[J]. arXiv preprint arXiv:1907.09595, 2019.
