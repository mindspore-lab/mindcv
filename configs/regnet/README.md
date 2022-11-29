# RegNet
> [Designing Network Design Spaces](https://arxiv.org/pdf/2003.13678.pdf)

## Introduction
***

In this work, we present a new network design paradigm that combines the advantages of manual design and NAS. Instead of focusing on designing individual network instances, we design design spaces that parametrize populations of networks. Like in manual design, we aim for interpretability and to discover general design principles that describe networks that are simple, work well, and generalize across settings. Like in NAS, we aim to take advantage of semi-automated procedures to help achieve these goals The general strategy we adopt is to progressively design simplified versions of an initial, relatively unconstrained, design space while maintaining or improving its quality. The overall process is analogous to manual design, elevated to the population level and guided via distribution estimates of network design spaces. As a testbed for this paradigm, our focus is on exploring network structure (e.g., width, depth, groups, etc.) assuming standard model families including VGG, ResNet, and ResNeXt. We start with a relatively unconstrained design space we call AnyNet (e.g., widths and depths vary freely across stages) and apply our humanin-the-loop methodology to arrive at a low-dimensional design space consisting of simple “regular” networks, that we call RegNet. The core of the RegNet design space is simple: stage widths and depths are determined by a quantized linear function. Compared to AnyNet, the RegNet design space has simpler models, is easier to interpret, and has a higher concentration of good models.

![](regnet.png)

## Results
***

| Model           | Context   |  Top-1 (%)  |  Params (M)    | Train T. | Infer T. |  Download | Config | Log |
|-----------------|-----------|-------|------------|-------|--------|---|--------|--------------|
| RegNetX-800MF | D910x8-G | 76.09     | 7.3       | 115s/epoch | 1.8ms/step | [model]() | [cfg]() | [log]() |

#### Notes

- All models are trained on ImageNet-1K training set and the top-1 accuracy is reported on the validatoin set.
- Context: GPU_TYPE x pieces - G/F, G - graph mode, F - pynative mode with ms function.  

## Quick Start
***
### Preparation

#### Installation
Please refer to the [installation instruction](https://github.com/mindspore-ecosystem/mindcv#installation) in MindCV.

#### Dataset Preparation
Please download the [ImageNet-1K](https://www.image-net.org/download.php) dataset for model training and validation.

### Training

- The yaml config files that yield competitive results on ImageNet for different models are listed in the configs folder. To trigger training using preset yaml config.

  ```shell
  python train.py --model=regnet_x_800mf --config=configs/regnet/regnet_x_800mf_ascend.yaml
  ```

### Validation

- To validate the trained model, you can use `validate.py`. Here is an example for regnetx800mf to verify the accuracy of
  pretrained weights.

  ```shell
  python validate.py --model=regnet_x_800mf --dataset=imagenet --val_split=val --pretrained
  ```

- To validate the model, you can use `validate.py`. Here is an example for regnetx800mf to verify the accuracy of your
  training.

  ```shell
  python validate.py --model=regnet_x_800mf --dataset=imagenet --val_split=val --ckpt_path='./ckpt/regnet_x_800mf-best.ckpt'
  ```

### Deployment (optional)

Please refer to the deployment tutorial in MindCV.


