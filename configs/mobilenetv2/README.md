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

## Results

Our reproduced model performance on ImageNet-1K is reported as follows.

<div align="center">

| Model                | Context  | Top-1 (%) | Top-5 (%) | Params (M) | Recipe                                                                                                     | Download                                                                                                         |
|----------------------|----------|-----------|-----------|------------|------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| mobilenet_v2_075 | D910x8-G | 69.98     | 89.32     | 2.66       | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/mobilenetv2/mobilenet_v2_0.75_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/mobilenet/mobilenetv2/mobilenet_v2_075-bd7bd4c4.ckpt) |
| mobilenet_v2_100 | D910x8-G | 72.27     | 90.72     | 3.54       | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/mobilenetv2/mobilenet_v2_1.0_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/mobilenet/mobilenetv2/mobilenet_v2_100-d5532038.ckpt) |
| mobilenet_v2_140 | D910x8-G | 75.56     | 92.56     | 6.15       | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/mobilenetv2/mobilenet_v2_1.4_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/mobilenet/mobilenetv2/mobilenet_v2_140-98776171.ckpt) |

</div>

#### Notes

- Context: Training context denoted as {device}x{pieces}-{MS mode}, where mindspore mode can be G - graph mode or F - pynative mode with ms function. For example, D910x8-G is for training on 8 pieces of Ascend 910 NPU using graph mode.
- Top-1 and Top-5: Accuracy reported on the validation set of ImageNet-1K.

## Quick Start

### Preparation

#### Installation
Please refer to the [installation instruction](https://github.com/mindspore-ecosystem/mindcv#installation) in MindCV.

#### Dataset Preparation
Please download the [ImageNet-1K](https://www.image-net.org/challenges/LSVRC/2012/index.php) dataset for model training and validation.

### Training

* Distributed Training

It is easy to reproduce the reported results with the pre-defined training recipe. For distributed training on multiple Ascend 910 devices, please run

```shell
# distributed training on multiple GPU/Ascend devices
mpirun -n 8 python train.py --config configs/mobilenetv2/mobilenet_v2_0.75_ascend.yaml --data_dir /path/to/imagenet
```

> If the script is executed by the root user, the `--allow-run-as-root` parameter must be added to `mpirun`.

Similarly, you can train the model on multiple GPU devices with the above `mpirun` command.

For detailed illustration of all hyper-parameters, please refer to [config.py](https://github.com/mindspore-lab/mindcv/blob/main/config.py).

**Note:**  As the global batch size  (batch_size x num_devices) is an important hyper-parameter, it is recommended to keep the global batch size unchanged for reproduction or adjust the learning rate linearly to a new global batch size.

* Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on a CPU/GPU/Ascend device
python train.py --config configs/mobilenetv2/mobilenet_v2_0.75_ascend.yaml --data_dir /path/to/dataset --distribute False
```

### Validation

To validate the accuracy of the trained model, you can use `validate.py` and parse the checkpoint path with `--ckpt_path`.

```shell
python validate.py -c configs/mobilenetv2/mobilenet_v2_0.75_ascend.yaml --data_dir /path/to/imagenet --ckpt_path /path/to/ckpt
```

### Deployment

Please refer to the [deployment tutorial](https://mindspore-lab.github.io/mindcv/tutorials/deployment/) in MindCV.

## References

[1] Sandler M, Howard A, Zhu M, et al. Mobilenetv2: Inverted residuals and linear bottlenecks[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2018: 4510-4520.
