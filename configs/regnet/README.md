# RegNet

> [Designing Network Design Spaces](https://arxiv.org/abs/2003.13678)

## Requirements
| mindspore | ascend driver |  firmware   | cann toolkit/kernel |
| :-------: | :-----------: | :---------: | :-----------------: |
|   2.3.1   |   24.1.RC2    | 7.3.0.1.231 |    8.0.RC2.beta1    |

## Introduction

In this work, the authors present a new network design paradigm that combines the advantages of manual design and NAS.
Instead of focusing on designing individual network instances, they design design spaces that parametrize populations of
networks. Like in manual design, the authors aim for interpretability and to discover general design principles that
describe networks that are simple, work well, and generalize across settings. Like in NAS, the authors aim to take
advantage of semi-automated procedures to help achieve these goals The general strategy they adopt is to progressively
design simplified versions of an initial, relatively unconstrained, design space while maintaining or improving its
quality. The overall process is analogous to manual design, elevated to the population level and guided via distribution
estimates of network design spaces. As a testbed for this paradigm, their focus is on exploring network structure (e.g.,
width, depth, groups, etc.) assuming standard model families including VGG, ResNet, and ResNeXt. The authors start with
a relatively unconstrained design space they call AnyNet (e.g., widths and depths vary freely across stages) and apply
their humanin-the-loop methodology to arrive at a low-dimensional design space consisting of simple “regular” networks,
that they call RegNet. The core of the RegNet design space is simple: stage widths and depths are determined by a
quantized linear function. Compared to AnyNet, the RegNet design space has simpler models, is easier to interpret, and
has a higher concentration of good models.[[1](#References)]

![RegNet](https://user-images.githubusercontent.com/74176172/210046899-4e83bb56-f7f6-49b2-9dde-dce200428e92.png)

## Performance

Our reproduced model performance on ImageNet-1K is reported as follows.

- Experiments are tested on ascend 910* with mindspore 2.3.1 graph mode

<div align="center">


| model name     | params(M) | cards | batch size | resolution | jit level | graph compile | ms/step | img/s    | acc@top1 | acc@top5 | recipe                                                                                              | weight                                                                                                     |
| -------------- | --------- | ----- | ---------- | ---------- | --------- | ------------- | ------- | -------- | -------- | -------- | --------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| regnet_x_800mf | 7.26      | 8     | 64         | 224x224    | O2        | 228s          | 50.74   | 10090.66 | 76.11    | 93.00    | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/regnet/regnet_x_800mf_ascend.yaml) | [weights](https://download-mindspore.osinfra.cn/toolkits/mindcv/regnet/regnet_x_800mf-68fe1cca-910v2.ckpt) |

</div>

- Experiments are tested on ascend 910 with mindspore 2.3.1 graph mode

<div align="center">


| model name     | params(M) | cards | batch size | resolution | jit level | graph compile | ms/step | img/s    | acc@top1 | acc@top5 | recipe                                                                                              | weight                                                                                       |
| -------------- | --------- | ----- | ---------- | ---------- | --------- | ------------- | ------- | -------- | -------- | -------- | --------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| regnet_x_800mf | 7.26      | 8     | 64         | 224x224    | O2        | 99s           | 42.49   | 12049.89 | 76.04    | 92.97    | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/regnet/regnet_x_800mf_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/regnet/regnet_x_800mf-617227f4.ckpt) |

</div>

#### Notes

- Top-1 and Top-5: Accuracy reported on the validation set of ImageNet-1K.

## Quick Start

### Preparation

#### Installation

Please refer to the [installation instruction](https://mindspore-lab.github.io/mindcv/installation/) in MindCV.

#### Dataset Preparation

Please download the [ImageNet-1K](https://www.image-net.org/challenges/LSVRC/2012/index.php) dataset for model training
and validation.

### Training

- Distributed Training

It is easy to reproduce the reported results with the pre-defined training recipe. For distributed training on multiple
Ascend 910 devices, please run

```shell
# distributed training on multiple NPU devices
msrun --bind_core=True --worker_num 8 python train.py --config configs/regnet/regnet_x_800mf_ascend.yaml --data_dir /path/to/imagenet
```



For detailed illustration of all hyper-parameters, please refer
to [config.py](https://github.com/mindspore-lab/mindcv/blob/main/config.py).

**Note:** As the global batch size (batch_size x num_devices) is an important hyper-parameter, it is recommended to keep
the global batch size unchanged for reproduction or adjust the learning rate linearly to a new global batch size.

* Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on single NPU device
python train.py --config configs/regnet/regnet_x_800mf_ascend.yaml --data_dir /path/to/imagenet --distribute False
```

### Validation

To validate the accuracy of the trained model, you can use `validate.py` and parse the checkpoint path
with `--ckpt_path`.

```shell
python validate.py --model=regnet_x_800mf --data_dir /path/to/imagenet --ckpt_path /path/to/ckpt
```


## References

[1]. Radosavovic I, Kosaraju R P, Girshick R, et al. Designing network design spaces[C]//Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition. 2020: 10428-10436.
