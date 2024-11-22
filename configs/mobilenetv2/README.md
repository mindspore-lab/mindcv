# MobileNetV2
> [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)

## Introduction

The model is a new neural network architecture that is specifically tailored for mobile and resource-constrained environments. This network pushes the state of the art for mobile custom computer vision models, significantly reducing the amount of operations and memory required while maintaining the same accuracy.

The main innovation of the model is the proposal of a new layer module: The Inverted Residual with Linear Bottleneck. The module takes as input a low-dimensional compressed representation that is first extended to high-dimensionality and then filtered with lightweight depth convolution. Linear convolution is then used to project the features back to the low-dimensional representation.[[1](#references)]

<p align="center">
  <img src="https://user-images.githubusercontent.com/53842165/210044190-8b5aab08-75fe-4e2c-87cc-d3529d9c60cd.png" width=800 />
</p>
<p align="center">
  <em>Figure 1. Architecture of MobileNetV2 [<a href="#references">1</a>] </em>
</p>

## Requirements
| mindspore | ascend driver |  firmware   | cann toolkit/kernel |
| :-------: | :-----------: | :---------: | :-----------------: |
|   2.3.1   |   24.1.RC2    | 7.3.0.1.231 |    8.0.RC2.beta1    |



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
msrun --bind_core=True --worker_num 8 python train.py --config configs/mobilenetv2/mobilenet_v2_0.75_ascend.yaml --data_dir /path/to/imagenet
```




For detailed illustration of all hyper-parameters, please refer to [config.py](https://github.com/mindspore-lab/mindcv/blob/main/config.py).

**Note:**  As the global batch size  (batch_size x num_devices) is an important hyper-parameter, it is recommended to keep the global batch size unchanged for reproduction or adjust the learning rate linearly to a new global batch size.

* Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on single NPU device
python train.py --config configs/mobilenetv2/mobilenet_v2_0.75_ascend.yaml --data_dir /path/to/dataset --distribute False
```

### Validation

To validate the accuracy of the trained model, you can use `validate.py` and parse the checkpoint path with `--ckpt_path`.

```shell
python validate.py -c configs/mobilenetv2/mobilenet_v2_0.75_ascend.yaml --data_dir /path/to/imagenet --ckpt_path /path/to/ckpt
```

## Performance

Our reproduced model performance on ImageNet-1K is reported as follows.

Experiments are tested on ascend 910* with mindspore 2.3.1 graph mode.


| model name       | params(M) | cards | batch size | resolution | jit level | graph compile | ms/step | img/s    | acc@top1 | acc@top5 | recipe                                                                                                      | weight                                                                                                                      |
| ---------------- | --------- | ----- | ---------- | ---------- | --------- |---------------| ------- | -------- | -------- | -------- | ----------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| mobilenet_v2_075 | 2.66      | 8     | 256        | 224x224    | O2        | 233s          | 174.65  | 11726.31 | 69.73    | 89.35    | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/mobilenetv2/mobilenet_v2_0.75_ascend.yaml) | [weights](https://download-mindspore.osinfra.cn/toolkits/mindcv/mobilenet/mobilenetv2/mobilenet_v2_075-755932c4-910v2.ckpt) |


Experiments are tested on ascend 910 with mindspore 2.3.1 graph mode.

| model name       | params(M) | cards | batch size | resolution | jit level | graph compile | ms/step | img/s    | acc@top1 | acc@top5 | recipe                                                                                                      | weight                                                                                                        |
| ---------------- | --------- | ----- | ---------- | ---------- | --------- | ------------- | ------- | -------- | -------- | -------- | ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| mobilenet_v2_075 | 2.66      | 8     | 256        | 224x224    | O2        | 164s          | 155.94  | 13133.26 | 69.98    | 89.32    | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/mobilenetv2/mobilenet_v2_0.75_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/mobilenet/mobilenetv2/mobilenet_v2_075-bd7bd4c4.ckpt) |

### Notes
- top-1 and top-5: Accuracy reported on the validation set of ImageNet-1K.

## References

[1] Sandler M, Howard A, Zhu M, et al. Mobilenetv2: Inverted residuals and linear bottlenecks[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2018: 4510-4520.
