# RegNet

> [Designing Network Design Spaces](https://arxiv.org/abs/2003.13678)

## Introduction

In this work, the authors present a new network design paradigm that combines the advantages of manual design and NAS. Instead of focusing on designing individual network instances, they design design spaces that parametrize populations of networks. Like in manual design, the authors aim for interpretability and to discover general design principles that describe networks that are simple, work well, and generalize across settings. Like in NAS, the authors aim to take advantage of semi-automated procedures to help achieve these goals The general strategy they adopt is to progressively design simplified versions of an initial, relatively unconstrained, design space while maintaining or improving its quality. The overall process is analogous to manual design, elevated to the population level and guided via distribution estimates of network design spaces. As a testbed for this paradigm, their focus is on exploring network structure (e.g., width, depth, groups, etc.) assuming standard model families including VGG, ResNet, and ResNeXt. The authors start with a relatively unconstrained design space they call AnyNet (e.g., widths and depths vary freely across stages) and apply their humanin-the-loop methodology to arrive at a low-dimensional design space consisting of simple “regular” networks, that they call RegNet. The core of the RegNet design space is simple: stage widths and depths are determined by a quantized linear function. Compared to AnyNet, the RegNet design space has simpler models, is easier to interpret, and has a higher concentration of good models.[[1](#References)]

![RegNet](https://user-images.githubusercontent.com/74176172/210046899-4e83bb56-f7f6-49b2-9dde-dce200428e92.png)

## Results

Our reproduced model performance on ImageNet-1K is reported as follows.

<div align="center">

|     Model      | Context  | Top-1 (%) | Top-5 (%) | Params (M) | Recipe                                                                                              | Download                                                                                     |
|:--------------:|:--------:|:---------:|:---------:|:----------:|-----------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|
| regnet_x_200mf | D910x8-G |   68.74   |   88.38   |    2.68    | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/regnet/regnet_x_200mf_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/regnet/regnet_x_200mf-0c2b1eb5.ckpt) |
| regnet_x_400mf | D910x8-G |   73.16   |   91.35   |    5.16    | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/regnet/regnet_x_400mf_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/regnet/regnet_x_400mf-4848837d.ckpt) |
| regnet_x_600mf | D910x8-G |   74.34   |   92.00   |    6.20    | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/regnet/regnet_x_600mf_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/regnet/regnet_x_600mf-ccd76c94.ckpt) |
| regnet_x_800mf | D910x8-G |   76.04   |   92.97   |    7.26    | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/regnet/regnet_x_800mf_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/regnet/regnet_x_800mf-617227f4.ckpt) |
| regnet_y_200mf | D910x8-G |   70.30   |   89.61   |    3.16    | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/regnet/regnet_y_200mf_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/regnet/regnet_y_200mf-76a2f720.ckpt) |
| regnet_y_400mf | D910x8-G |   73.91   |   91.84   |    4.34    | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/regnet/regnet_y_400mf_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/regnet/regnet_y_400mf-d496799d.ckpt) |
| regnet_y_600mf | D910x8-G |   75.69   |   92.50   |    6.06    | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/regnet/regnet_y_600mf_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/regnet/regnet_y_600mf-a84e19b2.ckpt) |
| regnet_y_800mf | D910x8-G |   76.52   |   93.10   |    6.26    | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/regnet/regnet_y_800mf_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/regnet/regnet_y_800mf-9b5211bd.ckpt) |
| regnet_y_16gf  | D910x8-G |   82.92   |   96.29   |   83.71    | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/regnet/regnet_y_16gf_ascend.yaml)  | [weights](https://download.mindspore.cn/toolkits/mindcv/regnet/regnet_y_16gf-c30a856f.ckpt)  |

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

- Distributed Training

It is easy to reproduce the reported results with the pre-defined training recipe. For distributed training on multiple Ascend 910 devices, please run

```shell
# distributed training on multiple GPU/Ascend devices
mpirun -n 8 python train.py --config configs/regnet/regnet_x_800mf_ascend.yaml --data_dir /path/to/imagenet
```
> If the script is executed by the root user, the `--allow-run-as-root` parameter must be added to `mpirun`.

Similarly, you can train the model on multiple GPU devices with the above `mpirun` command.

For detailed illustration of all hyper-parameters, please refer to [config.py](https://github.com/mindspore-lab/mindcv/blob/main/config.py).

**Note:** As the global batch size (batch_size x num_devices) is an important hyper-parameter, it is recommended to keep the global batch size unchanged for reproduction or adjust the learning rate linearly to a new global batch size.

* Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on a CPU/GPU/Ascend device
python train.py --config configs/regnet/regnet_x_800mf_ascend.yaml --data_dir /path/to/imagenet --distribute False
```

### Validation

To validate the accuracy of the trained model, you can use `validate.py` and parse the checkpoint path with `--ckpt_path`.

```shell
python validate.py --model=regnet_x_800mf --data_dir /path/to/imagenet --ckpt_path /path/to/ckpt
```

### Deployment

To deploy online inference services with the trained model efficiently, please refer to the [deployment tutorial](https://mindspore-lab.github.io/mindcv/tutorials/deployment/).

## References

[1]. Radosavovic I, Kosaraju R P, Girshick R, et al. Designing network design spaces[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020: 10428-10436.
