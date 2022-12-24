# EdgeNeXt

> [EdgeNeXt: Efficiently Amalgamated CNN-Transformer Architecture for Mobile Vision Applications](https://arxiv.org/abs/2206.10589)

## Introduction

***

In the pursuit of achieving ever-increasing accuracy, large and complex neural networks are usually developed. Such models demand high computational resources and therefore cannot be deployed on edge devices. It is of great interest to build resource-efficient general purpose networks due to their usefulness in several application areas. In this work, we strive to effectively combine the strengths of both CNN and Transformer models and propose a new efficient hybrid architecture EdgeNeXt. Specifically in EdgeNeXt, we introduce split depth-wise transpose attention (SDTA) encoder that splits input tensors into multiple channel groups and utilizes depth-wise convolution along with self-attention across channel dimensions to implicitly increase the receptive field and encode multi-scale features.

![](edgenext.png)

## Results

***

| Model           | Context   |  Top-1 (%)  | Top-5 (%)  |  Params (M)    | Train T. | Infer T. |  Download | Config | Log |
|-----------------|-----------|-------|-------|------------|-------|--------|---|--------|--------------|
| edgenext_small | D910x8-G | 79.146     | 94.394     | 5.59       | 518s/epoch | 238.6ms/step | [model](https://download.mindspore.cn/toolkits/mindcv/edgenext/edgenext_small.ckpt) | [cfg](https://github.com/mindspore-lab/mindcv/blob/main/configs/edgenext/edgenext_small_ascend.yaml) | [log]() |

#### Notes

- All models are trained on ImageNet-1K training set and the top-1 accuracy is reported on the validatoin set.
- Context: D910 x 8 - G, D910 - Ascend 910, x8 - 8 devices, G - graph mode.

## Quick Start

***

### Preparation

#### Installation

Please refer to the [installation instruction](https://github.com/mindspore-lab/mindcv#installation) in MindCV.

#### Dataset Preparation

Please download the [ImageNet-1K](https://www.image-net.org/download.php) dataset for model training and validation.

### Training

- **Hyper-parameters.** The hyper-parameter configurations for producing the reported results are stored in the yaml files in `mindcv/configs/edgenext  ` folder. For example, to train with one of these configurations, you can run:

  ```shell
  # train edgenext_small on 8 Ascends
  mpirun -n 8 python train.py -c configs/edgenext/edgenext_small_ascend.yaml --data_dir /path/to/imagenet_dir
  ```
  
  Note that the number of GPUs/Ascends and batch size will influence the training results. To reproduce the training result at most, it is recommended to use the **same number of GPUs/Ascends** with the same batch size.

Detailed adjustable parameters and their default value can be seen in [config.py](../../config.py).

### Validation

- To validate the trained model, you can use `validate.py`. Here is an example for edgenext_small to verify the accuracy of pretrained weights.

  ```shell
  python validate.py --model=edgenext_small --data_dir=imagenet_dir --val_split=val --ckpt_path
  ```

### Deployment (optional)

Please refer to the deployment tutorial in MindCV.
