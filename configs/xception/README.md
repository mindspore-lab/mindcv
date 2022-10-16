# Xception

***
> [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/pdf/1610.02357.pdf)

## Introduction

***
Xception is another improved network of InceptionV3 in addition to inceptionV4, using a deep convolutional neural
network architecture involving depthwise separable convolution, which was developed by Google researchers. Google
interprets the Inception module in convolutional neural networks as an intermediate step between regular convolution and
depthwise separable convolution operations. From this point of view, the depthwise separable convolution can be
understood as having the largest number of Inception modules, that is, the extreme idea proposed in the paper, combined
with the idea of residual network, Google proposed a new type of deep convolutional neural network inspired by Inception
Network architecture where the Inception module has been replaced by a depthwise separable convolution module.

![](./Xception.jpg)

## Benchmark

***

|        |          |           |           |    Pynative     |  Pynative  |     Graph      |   Graph    |           |            |
| :----: | -------- | :-------: | :-------: | :-------------: | :--------: | :------------: | :--------: | :-------: | :--------: |
|        | Model    | Top-1 (%) | Top-5 (%) | train (s/epoch) | Infer (ms) | train(s/epoch) | Infer (ms) | Download  |   Config   |
|  GPU   | xception |           |           |                 |            |                |            | [model]() | [config]() |
| Ascend | xception |           |           |                 |            |                |            |           |            |

## Examples

***

### Train

- The [yaml config files](../../configs) that yield competitive results on ImageNet for different models are listed in
  the `configs` folder. To trigger training using preset yaml config.

  ```shell
  comming soon
  ```

- Here is the example for finetuning a pretrained InceptionV3 on CIFAR10 dataset using Momentum optimizer.

  ```shell
  python train.py --model=xception --pretrained --opt=momentum --lr=0.001 dataset=cifar10 --num_classes=10 --dataset_download
  ```

Detailed adjustable parameters and their default value can be seen in [config.py](../../config.py).

### Eval

- To validate the model, you can use `validate.py`. Here is an example to verify the accuracy of pretrained weights.

  ```shell
  python validate.py --model=xception --dataset=imagenet --val_split=val --pretrained
  ```

- To validate the model, you can use `validate.py`. Here is an example to verify the accuracy of your training.

  ```shell
  python validate.py --model=xception --dataset=imagenet --val_split=val --ckpt_path='./ckpt/xception-best.ckpt'
  ```
