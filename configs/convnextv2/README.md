# ConvNeXt V2
> [ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders](https://arxiv.org/abs/2301.00808)


## Introduction

In this paper, the authors propose a fully convolutional masked autoencoder framework and a new Global Response
Normalization (GRN) layer that can be added to the ConvNeXt architecture to enhance inter-channel feature competition.
This co-design of self-supervised learning techniques (such as MAE) and architectural improvement results in a new model
family called ConvNeXt V2, which significantly improves the performance of pure ConvNets on various recognition
benchmarks, including ImageNet classification, COCO detection, and ADE20K segmentation.[[1](#references)]

<p align="center">
  <img src="https://github.com/facebookresearch/ConvNeXt-V2/assets/53842165/d7dbd994-0577-42e3-9068-67d32b8a3bcb" width=350 />
</p>
<p align="center">
  <em>Figure 1. Architecture of ConvNeXt V2 [<a href="#references">1</a>] </em>
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
msrun --bind_core=True --worker_num 8 python train.py --config configs/convnextv2/convnextv2_tiny_ascend.yaml --data_dir /path/to/imagenet
```



For detailed illustration of all hyper-parameters, please refer to [config.py](https://github.com/mindspore-lab/mindcv/blob/main/config.py).

**Note:**  As the global batch size  (batch_size x num_devices) is an important hyper-parameter, it is recommended to keep the global batch size unchanged for reproduction or adjust the learning rate linearly to a new global batch size.

* Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on single NPU device
python train.py --config configs/convnextv2/convnextv2_tiny_ascend.yaml --data_dir /path/to/dataset --distribute False
```

### Validation

To validate the accuracy of the trained model, you can use `validate.py` and parse the checkpoint path with `--ckpt_path`.

```shell
python validate.py -c configs/convnextv2/convnextv2_tiny_ascend.yaml --data_dir /path/to/imagenet --ckpt_path /path/to/ckpt
```

## Performance

Our reproduced model performance on ImageNet-1K is reported as follows.

Experiments are tested on ascend 910* with mindspore 2.5.0 graph mode.

| model name      | params(M) | cards | batch size | resolution | jit level | graph compile | ms/step | img/s   | acc@top1 | acc@top5 | recipe                                                                                                   | weight                                                                                                          |
| --------------- | --------- | ----- | ---------- | ---------- | --------- | ------------- |---------| ------- | -------- | -------- | -------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| convnextv2_tiny | 28.64     | 8     | 128        | 224x224    | O2        | 268s          | 280.47  | 3651.01 | 82.39    | 95.95    | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/convnextv2/convnextv2_tiny_ascend.yaml) | [weights](https://download-mindspore.osinfra.cn/toolkits/mindcv/convnextv2/convnextv2_tiny-a35b79ce-910v2.ckpt) |

### Notes
- top-1 and top-5: Accuracy reported on the validation set of ImageNet-1K.

## References

[1] Woo S, Debnath S, Hu R, et al. ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders[J]. arXiv preprint arXiv:2301.00808, 2023.
