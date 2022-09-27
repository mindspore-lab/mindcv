# MobileNetV1
***
> [MobileNetV1: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/pdf/1704.04861.pdf)

## Introduction
***
Compared with the traditional convolutional neural network, MobileNetV1's parameters and the amount of computation are greatly reduced on the premise that the accuracy rate is slightly reduced.
(Compared to VGG16, the accuracy rate is reduced by 0.9%, but the model parameters are only 1/32 of VGG). The model is based on a streamlined architecture that uses depthwise separable convolutions to build lightweight deep neural networks.
At the same time, two simple global hyperparameters are introduced, which can effectively trade off latency and accuracy.


## Benchmark
***

|        |           |           |           |    Pynative     |  Pynative  |     Graph      |   Graph    |           |            |
| :----: | --------- | :-------: | :-------: | :-------------: | :--------: | :------------: | :--------: | :-------: | :--------: |
|        | Model     | Top-1 (%) | Top-5 (%) | train (s/epoch) | Infer (ms) | train(s/epoch) | Infer (ms) | Download  |   Config   |
|  GPU   | MobileNet_v1_100 | 71.95     | 90.41     |                 |            |                |            | [model]() | [config]() |
| Ascend | MobileNet_v1_100 | 71.83     | 90.26     |                 |            |                |            |           |            |
|  GPU   | MobileNet_v1_075 | 70.84     | 89.63     |                 |            |                |            | [model]() | [config]() |
| Ascend | MobileNet_v1_075 | 70.66     | 89.49     |                 |            |                |            |           |            |
|  GPU   | MobileNet_v1_050 | 66.37     | 86.71     |                 |            |                |            | [model]() | [config]() |
| Ascend | MobileNet_v1_050 | 66.39     | 86.85     |                 |            |                |            |           |            |
|  GPU   | MobileNet_v1_025 | 54.58     | 78.27     |                 |            |                |            | [model]() | [config]() |
| Ascend | MobileNet_v1_025 | 54.64     | 78.29     |                 |            |                |            |           |            |

## Examples

***

### Train

- The [yaml config files](../../config) that yield competitive results on ImageNet for different models are listed in the `config` folder. To trigger training using preset yaml config. 

  ```shell
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  mpirun --allow-run-as-root -n 8 python train.py -c config/mobilenetv1/mobilenetv1_075_gpu.yaml
  ```


- Here is the example for finetuning a pretrained MobileNetV1 on CIFAR10 dataset using Adam optimizer.

  ```shell
  python train.py --model=mobilenet_v1_075_224 --pretrained --opt=adam --lr=0.001 ataset=cifar10 --num_classes=10 --dataset_download
  ```
  
  Detailed adjustable parameters and their default value can be seen in [config.py](../../config.py)

### Eval

- To validate the model, you can use `validate.py`. Here is an example to verify the accuracy of pretrained weights.

  ```shell
  python validate.py --model=mobilenet_v1_075_224 --dataset=imagenet --val_split=val --pretrained
  ```

- To validate the model, you can use `validate.py`. Here is an example to verify the accuracy of your training.

  ```shell
  python validate.py --model=mobilenet_v1_075_224 --dataset=imagenet --val_split=val --ckpt_path='./ckpt/mobilenet_v1_075_224-best.ckpt' 
  ```

