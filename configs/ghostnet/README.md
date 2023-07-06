# GhostNet
> [GhostNet: More Features from Cheap Operations](https://arxiv.org/abs/1911.11907)

## Introduction

The redundancy in feature maps is an important characteristic of those successful CNNs, but has rarely been
investigated in neural architecture design. This paper proposes a novel Ghost module to generate more feature maps from
cheap operations. Based on a set of intrinsic feature maps, the authors apply a series of linear transformations with
cheap cost to generate many ghost feature maps that could fully reveal information underlying intrinsic features. The
proposed Ghost module can be taken as a plug-and-play component to upgrade existing convolutional neural networks.
Ghost bottlenecks are designed to stack Ghost modules, and then the lightweight GhostNet can be easily
established. Experiments conducted on benchmarks demonstrate that the Ghost module is an impressive alternative of
convolution layers in baseline models, and GhostNet can achieve higher recognition performance (e.g. 75.7% top-1
accuracy) than MobileNetV3 with similar computational cost on the ImageNet ILSVRC-2012 classification
dataset.[[1](#references)]

<p align="center">
  <img src="https://user-images.githubusercontent.com/53842165/230816651-8466df07-dddc-4a42-9a2d-743e8f2fdad3.png" width=500 />
</p>
<p align="center">
  <em>Figure 1. Architecture of GhostNet [<a href="#references">1</a>] </em>
</p>

## Results

Our reproduced model performance on ImageNet-1K is reported as follows.

<div align="center">

| Model        | Context  | Top-1 (%) | Top-5 (%) | Params (M) | Recipe                                                                                              | Download                                                                                     |
|--------------|----------|-----------|-----------|------------|-----------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|
| ghostnet_050 | D910x8-G | 66.03     | 86.64     | 2.60       | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/ghostnet/ghostnet_050_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/ghostnet/ghostnet_050-85b91860.ckpt) |
| ghostnet_100 | D910x8-G | 73.78     | 91.66     | 5.20       | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/ghostnet/ghostnet_100_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/ghostnet/ghostnet_100-bef8025a.ckpt) |
| ghostnet_130 | D910x8-G | 75.50     | 92.56     | 7.39       | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/ghostnet/ghostnet_130_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/ghostnet/ghostnet_130-cf4c235c.ckpt) |

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
mpirun -n 8 python train.py --config configs/ghostnet/ghostnet_100_ascend.yaml --data_dir /path/to/imagenet
```

> If the script is executed by the root user, the `--allow-run-as-root` parameter must be added to `mpirun`.

Similarly, you can train the model on multiple GPU devices with the above `mpirun` command.

For detailed illustration of all hyper-parameters, please refer to [config.py](https://github.com/mindspore-lab/mindcv/blob/main/config.py).

**Note:**  As the global batch size  (batch_size x num_devices) is an important hyper-parameter, it is recommended to keep the global batch size unchanged for reproduction or adjust the learning rate linearly to a new global batch size.

* Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on a CPU/GPU/Ascend device
python train.py --config configs/ghostnet/ghostnet_100_ascend.yaml --data_dir /path/to/dataset --distribute False
```

### Validation

To validate the accuracy of the trained model, you can use `validate.py` and parse the checkpoint path with `--ckpt_path`.

```shell
python validate.py -c configs/ghostnet/ghostnet_100_ascend.yaml --data_dir /path/to/imagenet --ckpt_path /path/to/ckpt
```

### Deployment

Please refer to the [deployment tutorial](https://mindspore-lab.github.io/mindcv/tutorials/deployment/) in MindCV.

## References

[1] Han K, Wang Y, Tian Q, et al. Ghostnet: More features from cheap operations[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020: 1580-1589.
