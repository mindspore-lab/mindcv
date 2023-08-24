# ResNet

> [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

## Introduction

Resnet is a residual learning framework to ease the training of networks that are substantially deeper than those used previously, which is explicitly reformulated that the layers are learning residual functions with reference to the layer inputs, instead of learning unreferenced functions. Lots of comprehensive empirical evidence showing that these residual networks are easier to optimize, and can gain accuracy from considerably increased depth.

<p align="center">
  <img src="https://user-images.githubusercontent.com/121591093/210049796-79b70294-4250-4d0f-b078-2471880884cf.png" width=800 />
</p>
<p align="center">
  <em>Figure 1. Architecture of ResNet [<a href="#references">1</a>] </em>
</p>

## Results

Our reproduced model performance on ImageNet-1K is reported as follows.

<div align="center">

| Model           | Context   | Top-1 (%) | Top-5 (%) |  Params (M) | Recipe                                                                                          | Download                                                                           |
|-----------------|-----------|-----------|-----------|-------|-------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------|
| resnet18 | D910x8-G | 70.21     | 89.62     |  11.70   | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/resnet/resnet_18_ascend.yaml)  | [weights](https://download.mindspore.cn/toolkits/mindcv/resnet/resnet18-1e65cd21.ckpt)  |
| resnet34 | D910x8-G | 74.15     | 91.98     | 21.81  | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/resnet/resnet_34_ascend.yaml)  | [weights](https://download.mindspore.cn/toolkits/mindcv/resnet/resnet34-f297d27e.ckpt)  |
| resnet50 | D910x8-G | 76.69     | 93.50     | 25.61 | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/resnet/resnet_50_ascend.yaml)  | [weights](https://download.mindspore.cn/toolkits/mindcv/resnet/resnet50-e0733ab8.ckpt)  |
| resnet101 | D910x8-G | 78.24     | 94.09     |44.65 | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/resnet/resnet_101_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/resnet/resnet101-689c5e77.ckpt) |
| resnet152 | D910x8-G | 78.72     | 94.45     | 60.34| [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/resnet/resnet_152_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/resnet/resnet152-beb689d8.ckpt) |

</div>

#### Notes
- Context: Training context denoted as {device}x{pieces}-{MS mode}, where mindspore mode can be G - graph mode or F - pynative mode with ms function. For example, D910x8-G is for training on 8 pieces of Ascend 910 NPU using graph mode.
- Top-1 and Top-5: Accuracy reported on the validation set of ImageNet-1K.


## Quick Start

### Preparation

#### Installation
Please refer to the [installation instruction](https://github.com/mindspore-lab/mindcv#installation) in MindCV.

#### Dataset Preparation
Please download the [ImageNet-1K](https://www.image-net.org/challenges/LSVRC/2012/index.php) dataset for model training and validation.

### Training

* Distributed Training

It is easy to reproduce the reported results with the pre-defined training recipe. For distributed training on multiple Ascend 910 devices, please run

```shell
# distributed training on multiple GPU/Ascend devices
mpirun -n 8 python train.py --config configs/resnet/resnet_18_ascend.yaml --data_dir /path/to/imagenet
```

> If the script is executed by the root user, the `--allow-run-as-root` parameter must be added to `mpirun`.

Similarly, you can train the model on multiple GPU devices with the above `mpirun` command.

For detailed illustration of all hyper-parameters, please refer to [config.py](https://github.com/mindspore-lab/mindcv/blob/main/config.py).

**Note:**  As the global batch size  (batch_size x num_devices) is an important hyper-parameter, it is recommended to keep the global batch size unchanged for reproduction or adjust the learning rate linearly to a new global batch size.

* Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on a CPU/GPU/Ascend device
python train.py --config configs/resnet/resnet_18_ascend.yaml --data_dir /path/to/dataset --distribute False
```

### Validation

To validate the accuracy of the trained model, you can use `validate.py` and parse the checkpoint path with `--ckpt_path`.

```shell
python validate.py -c configs/resnet/resnet_18_ascend.yaml --data_dir /path/to/imagenet --ckpt_path /path/to/ckpt
```

### Deployment

To deploy online inference services with the trained model efficiently, please refer to the [deployment tutorial](https://mindspore-lab.github.io/mindcv/tutorials/deployment/).

## References

[1] He K, Zhang X, Ren S, et al. Deep residual learning for image recognition[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2016: 770-778.
