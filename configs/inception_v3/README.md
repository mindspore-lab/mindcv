# InceptionV3

***

> [InceptionV3: Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/pdf/1512.00567.pdf)

## Introduction

***
InceptionV3 is an upgraded version of GoogleNet. One of the most important improvements of V3 is Factorization, which
decomposes 7x7 into two one-dimensional convolutions (1x7, 7x1), and 3x3 is the same (1x3, 3x1), such benefits, both It
can accelerate the calculation (excess computing power can be used to deepen the network), and can split 1 conv into 2
convs, which further increases the network depth and increases the nonlinearity of the network. It is also worth noting
that the network input from 224x224 has become 299x299, and 35x35/17x17/8x8 modules are designed more precisely. In
addition, V3 also adds batch normalization, which makes the model converge more quickly, which plays a role in partial
regularization and effectively reduces overfitting.
![](InceptionV3网络.jpg)

## Benchmark

***

|        |              |           |           |    Pynative     |  Pynative  |     Graph      |   Graph    |           |            |
| :----: | ------------ | :-------: | :-------: | :-------------: | :--------: | :------------: | :--------: | :-------: | :--------: |
|        | Model        | Top-1 (%) | Top-5 (%) | train (s/epoch) | Infer (ms) | train(s/epoch) | Infer (ms) | Download  |   Config   |
|  GPU   | inception_v3 |           |           |    1145.248     |            |    1063.01     |            | [model]() | [config]() |
| Ascend | inception_v3 |           |           |                 |            |                |            |           |            |

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
  python train.py --model=inception_v3 --pretrained --opt=momentum --lr=0.001 dataset=cifar10 --num_classes=10 --dataset_download
  ```

Detailed adjustable parameters and their default value can be seen in [config.py](../../config.py).

### Eval

- To validate the model, you can use `validate.py`. Here is an example to verify the accuracy of pretrained weights.

  ```shell
  python validate.py --model=inception_v3 --dataset=imagenet --val_split=val --pretrained
  ```

- To validate the model, you can use `validate.py`. Here is an example to verify the accuracy of your training.

  ```shell
  python validate.py --model=inception_v3 --dataset=imagenet --val_split=val --ckpt_path='./ckpt/inception_v3-best.ckpt'
  ```
