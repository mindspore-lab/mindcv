# MobileNetV3
***
> [MobileNetV3: Searching for MobileNetV3](https://arxiv.org/pdf/1512.00567.pdf)

## Introduction
***
The goal of designing MobileNetV3 is to develop the best mobile computer vision architecture to optimize accuracy and latency on mobile devices.
The highlights of this model are:

1) Complementary search technology;

2) a new efficient nonlinear version for mobile environments;

3) A new efficient network design idea;

4) New efficient segmentation decoder.

Extensive experiments demonstrate the efficacy and value of MobileNetV3 on a wide range of use cases and mobile phones


## Benchmark
***

|        |           |           |           |    Pynative     |  Pynative  |     Graph      |   Graph    |           |            |
| :----: | --------- | :-------: | :-------: | :-------------: | :--------: | :------------: | :--------: | :-------: | :--------: |
|        | Model     | Top-1 (%) | Top-5 (%) | train (s/epoch) | Infer (ms) | train(s/epoch) | Infer (ms) | Download  |   Config   |
|  GPU   | MobileNet_v3_large | 74.56     | 91.79     |                 |            |                |            | [model]() | [config]() |
| Ascend | MobileNet_v3_large | 74.61     | 91.82     |                 |            |                |            |           |            |
|  GPU   | MobileNet_v3_small | 67.46     | 87.07     |                 |            |                |            | [model]() | [config]() |
| Ascend | MobileNet_v3_small | 67.49     | 87.13     |                 |            |                |            |           |            |

## Examples

***

### Train

- The [yaml config files](../../config) that yield competitive results on ImageNet for different models are listed in the `config` folder. To trigger training using preset yaml config. 

  ```shell
  comming soon
  ```


- Here is the example for finetuning a pretrained MobileNetV3 on CIFAR10 dataset using Adam optimizer.

  ```shell
  python train.py --model=mobilenet_v3_large_100 --pretrained --opt=adam --lr=0.001 ataset=cifar10 --num_classes=10 --dataset_download
  ```
  
  Detailed adjustable parameters and their default value can be seen in [config.py](../../config.py)

### Eval

- To validate the model, you can use `validate.py`. Here is an example to verify the accuracy of pretrained weights.

  ```shell
  python validate.py --model=mobilenet_v3_large_100 --dataset=imagenet --val_split=val --pretrained
  ```

- To validate the model, you can use `validate.py`. Here is an example to verify the accuracy of your training.

  ```shell
  python validate.py --model=mobilenet_v3_large_100 --dataset=imagenet --val_split=val --ckpt_path='./ckpt/mobilenet_v3_large_100-best.ckpt' 
  ```

