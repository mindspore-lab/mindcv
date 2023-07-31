# SSD Based on MindCV Backbones

> [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325)

## Introduction

SSD is an single-staged object detector. It discretizes the output space of bounding boxes into a set of default boxes over different aspect ratios and scales per feature map location, and combines predictions from multi-scale feature maps to detect objects with various sizes. At prediction time, SSD generates scores for the presence of each object category in each default box and produces adjustments to the box to better match the object shape.

In this example, by leveraging [the multi-scale feature extraction of MindCV](https://github.com/mindspore-lab/mindcv/blob/main/docs/en/how_to_guides/feature_extraction.md), we demonstrate that using backbones from MindCV much simplifies the implementation of SSD.

## Configurations

Here, we provide three configurations of SSD.
* Using [MobileNetV2](https://github.com/mindspore-lab/mindcv/tree/main/configs/mobilenetv2) as the backbone and the original detector described in the paper.
* Using [ResNet50](https://github.com/mindspore-lab/mindcv/tree/main/configs/resnet) as the backbone with a FPN and a shared-weight-based detector.
* Using [MobileNetV3](https://github.com/mindspore-lab/mindcv/tree/main/configs/mobilenetv3) as the backbone and the original detector described in the paper.

## Dataset

We train and test SSD using [COCO 2017 Dataset](https://cocodataset.org/#download). The dataset contains
* 118000 images about 18 GB for training, and
* 5000 images about 1 GB for testing.

## Quick Start

### Preparation

Install dependencies as shown [here](https://mindspore-lab.github.io/mindcv/installation/).

Download [COCO 2017 Dataset](https://cocodataset.org/#download), prepare the dataset as follows, and specify the dataset path at keyword `data_dir` in the config file at.
```
.
└─cocodataset
  ├─annotations
    ├─instance_train2017.json
    └─instance_val2017.json
  ├─val2017
  └─train2017
```

Download the pretrained backbone weights from the table below, and specify the path of the backbone weights at keyword `backbone_ckpt_path` in the config file.
<div align="center">

|    MobileNetV2   |     ResNet50     |    MobileNetV3   |
|:----------------:|:----------------:|:----------------:|
| [backbone wegihts](https://download.mindspore.cn/toolkits/mindcv/mobilenet/mobilenetv2/mobilenet_v2_100-d5532038.ckpt) | [backbone wegihts](https://download.mindspore.cn/toolkits/mindcv/resnet/resnet50-e0733ab8.ckpt) | [backbone wegihts](https://download.mindspore.cn/toolkits/mindcv/mobilenet/mobilenetv3/mobilenet_v3_large_100-1279ad5f.ckpt) |

</div>

### Training

It is highly recommended to use distributed training for this SSD implementation.

For using MPI
