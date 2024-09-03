# MobileNetV1
> [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)

## Requirements
| mindspore | ascend driver |  firmware   | cann toolkit/kernel |
| :-------: | :-----------: | :---------: | :-----------------: |
|   2.3.1   |   24.1.RC2    | 7.3.0.1.231 |    8.0.RC2.beta1    |

## Introduction

Compared with the traditional convolutional neural network, MobileNetV1's parameters and the amount of computation are greatly reduced on the premise that the accuracy rate is slightly reduced. (Compared to VGG16, the accuracy rate is reduced by 0.9%, but the model parameters are only 1/32 of VGG). The model is based on a streamlined architecture that uses depthwise separable convolutions to build lightweight deep neural networks. At the same time, two simple global hyperparameters are introduced, which can effectively trade off latency and accuracy.[[1](#references)]

<p align="center">
  <img src="https://user-images.githubusercontent.com/53842165/210044118-20dcc78a-96cd-4a3f-9cd1-fefa00c12227.png" width=400 />
</p>
<p align="center">
  <em>Figure 1. Architecture of MobileNetV1 [<a href="#references">1</a>] </em>
</p>

## Performance

Our reproduced model performance on ImageNet-1K is reported as follows.

- Experiments are tested on ascend 910* with mindspore 2.3.1 graph mode

<div align="center">


| model name       | params(M) | cards | batch size | resolution | jit level | graph compile | ms/step | img/s    | acc@top1 | acc@top5 | recipe                                                                                                      | weight                                                                                                                      |
| ---------------- | --------- | ----- | ---------- | ---------- | --------- |---------------| ------- | -------- | -------- | -------- | ----------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| mobilenet_v1_025 | 0.47      | 8     | 64         | 224x224    | O2        | 195s          | 47.47   | 10785.76 | 54.05    | 77.74    | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/mobilenetv1/mobilenet_v1_0.25_ascend.yaml) | [weights](https://download-mindspore.osinfra.cn/toolkits/mindcv/mobilenet/mobilenetv1/mobilenet_v1_025-cbe3d3b3-910v2.ckpt) |


</div>

- Experiments are tested on ascend 910 with mindspore 2.3.1 graph mode

<div align="center">


| model name       | params(M) | cards | batch size | resolution | jit level | graph compile | ms/step | img/s    | acc@top1 | acc@top5 | recipe                                                                                                      | weight                                                                                                        |
| ---------------- | --------- | ----- | ---------- | ---------- | --------- | ------------- | ------- | -------- | -------- | -------- | ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| mobilenet_v1_025 | 0.47      | 8     | 64         | 224x224    | O2        | 89s           | 42.43   | 12066.93 | 53.87    | 77.66    | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/mobilenetv1/mobilenet_v1_0.25_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/mobilenet/mobilenetv1/mobilenet_v1_025-d3377fba.ckpt) |


</div>

#### Notes
- Top-1 and Top-5: Accuracy reported on the validation set of ImageNet-1K.

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
msrun --bind_core=True --worker_num 8 python train.py --config configs/mobilenetv1/mobilenet_v1_0.25_ascend.yaml --data_dir /path/to/imagenet
```




For detailed illustration of all hyper-parameters, please refer to [config.py](https://github.com/mindspore-lab/mindcv/blob/main/config.py).

**Note:**  As the global batch size  (batch_size x num_devices) is an important hyper-parameter, it is recommended to keep the global batch size unchanged for reproduction or adjust the learning rate linearly to a new global batch size.

* Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on single NPU device
python train.py --config configs/mobilenetv1/mobilenet_v1_0.25_ascend.yaml --data_dir /path/to/dataset --distribute False
```

### Validation

To validate the accuracy of the trained model, you can use `validate.py` and parse the checkpoint path with `--ckpt_path`.

```shell
python validate.py -c configs/mobilenetv1/mobilenet_v1_0.25_ascend.yaml --data_dir /path/to/imagenet --ckpt_path /path/to/ckpt
```


## References

[1] Howard A G, Zhu M, Chen B, et al. Mobilenets: Efficient convolutional neural networks for mobile vision applications[J]. arXiv preprint arXiv:1704.04861, 2017.
