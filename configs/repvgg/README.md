# RepVGG

***
> [RepVGG: Making VGG-style ConvNets Great Again](https://arxiv.org/pdf/2101.03697.pdf)

## Introduction

***
RepVGG, a vgg-style architecture that outperforms many complex models

Its main highlights are:

1) The model has a normal (a.k.a. feedforward) structure like vgg, without any other branches, each layer takes the
   output of its only previous layer as input, and feeds the output to its only next layer.

2) The body of the model uses only 3 × 3 conv and ReLU.

3) The specific architecture (including specific depth and layer width) is instantiated without automatic search, manual
   refinement, compound scaling, and other complicated designs.

## Benchmark

***

|        |           |           |           |    Pynative     |  Pynative  |     Graph      |   Graph    |           |            |
| :----: | --------- | :-------: | :-------: | :-------------: | :--------: | :------------: | :--------: | :-------: | :--------: |
|        | Model     | Top-1 (%) | Top-5 (%) | train (s/epoch) | Infer (ms) | train(s/epoch) | Infer (ms) | Download  |   Config   |
|  GPU   | RepVGG_A0 | 71.98     | 90.36     |                 |            |                |            | [model]() | [config]() |
| Ascend | RepVGG_A0 | 71.87     | 90.43     |                 |            |                |            |           |            |

## Examples

***

### Train

- The [yaml config files](../../configs) that yield competitive results on ImageNet for different models are listed in
  the `configs` folder. To trigger training using preset yaml config.

  ```shell
  comming soon
  ```

- Here is the example for finetuning a pretrained RepVGG_A0 on CIFAR10 dataset using Adam optimizer.

  ```shell
  python train.py --model=RepVGG_A0 --pretrained --opt=adam --lr=0.001 ataset=cifar10 --num_classes=10 --dataset_download
  ```

  Detailed adjustable parameters and their default value can be seen in [config.py](../../config.py)

### Eval

- To validate the model, you can use `validate.py`. Here is an example to verify the accuracy of pretrained weights.

  ```shell
  python validate.py --model=RepVGG_A0 --dataset=imagenet --val_split=val --pretrained
  ```

- To validate the model, you can use `validate.py`. Here is an example to verify the accuracy of your training.

  ```shell
  python validate.py --model=RepVGG_A0 --dataset=imagenet --val_split=val --ckpt_path='./ckpt/RepVGG_A0-best.ckpt' 
  ```
