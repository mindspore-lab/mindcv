# MobileNetV2

***
> [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/pdf/1801.04381.pdf)

## Introduction

***

The model is a new neural network architecture that is specifically tailored for mobile and resource-constrained environments.
This network pushes the state of the art for mobile custom computer vision models, significantly reducing the amount of operations and memory required while maintaining the same accuracy.

The main innovation of the model is the proposal of a new layer module: The Inverted Residual with Linear Bottleneck. The module takes as input a low-dimensional compressed representation that is first extended to high-dimensionality and then filtered with lightweight depth convolution.
Linear convolution is then used to project the features back to the low-dimensional representation.

![](mobilenetv2.png)

## Results
***

| Model           | Context   |  Top-1 (%)  | Top-5 (%)  |  Params (M)    | Train T. | Infer T. |  Download | Config | Log |
|-----------------|-----------|-------------|------------|----------------|----------|----------|-----------|--------|--------------|
| MobileNet_v2_075 | D910x8-G | 69.76       | 89.28      | 2.66           | 106s/epoch |        | [model]() | [cfg]() | [log]() |
| MobileNet_v2_100 | D910x8-G | 72.02       | 90.92      | 3.54           | 98s/epoch |        | [model]() | [cfg]() | [log]() |
| MobileNet_v2_140 | D910x8-G | 74.97       | 92.32      | 6.15           | 157s/epoch |        | [model]() | [cfg]() | [log]() |

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

- **Hyper-parameters.** The hyper-parameter configurations for producing the reported results are stored in the yaml files in `mindcv/configs/mobilenetv1` folder. For example, to train with one of these configurations, you can run:

  ```shell
  # train mobilenetv2 on 8 GPUs
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  mpirun -n 8 python train.py -c configs/mobilenetv2/mobilenetv2_100_gpu.yaml --data_dir /path/to/imagenet
  ```
  
  Note that the number of GPUs/Ascends and batch size will influence the training results. To reproduce the training result at most, it is recommended to use the **same number of GPUs/Ascneds** with the same batch size.

- **Finetuning.** Here is an example for finetuning a pretrained mobilenetv2_100 on CIFAR10 dataset using Momentum optimizer.

  ```shell
  python train.py -c configs/mobilenetv2/mobilenetv2_100_gpu.yaml --data_dir /path/to/imagenet
  ```

Detailed adjustable parameters and their default value can be seen in [config.py](../../config.py).

### Validation

- To validate the trained model, you can use `validate.py`. Here is an example for mobilenet_100 to verify the accuracy of
  pretrained weights.

  ```shell
  python validate.py -c /path/to/val.yaml --data_dir /path/to/imagenet
  ```

- To validate the model, you can use `validate.py`. Here is an example for mobilenetv1_100 to verify the accuracy of your
  training.

  ```shell
  python validate.py -c /path/to/val.yaml --data_dir /path/to/imagenet --dataset=imagenet --val_split=val --ckpt_path='./ckpt/mobilenet_v2_100_224-200_625.ckpt'
  ```

### Deployment (optional)

Please refer to the deployment tutorial in MindCV.
