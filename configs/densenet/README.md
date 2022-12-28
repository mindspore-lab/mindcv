# DenseNet
> [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)

## Introduction

Recent work has shown that convolutional networks can be substantially deeper, more accurate, and more efficient to train if
they contain shorter connections between layers close to the input and those close to the output. Dense Convolutional
Network (DenseNet) is introduced based on this observation, which connects each layer to every other layer in a
feed-forward fashion. Whereas traditional convolutional networks with $L$ layers have $L$ connections-one between each
layer and its subsequent layer, our network has $\frac{L(L+1)}{2}$ direct connections. For each layer, the feature maps
of all preceding layers are used as inputs, and their feature maps are used as inputs into all subsequent layers.
DenseNets have several compelling advantages: they alleviate the vanishing-gradient problem, strengthen feature
propagation, encourage feature reuse, and substantially reduce the number of parameters.

<p align="center">
  <img src="https://github.com/mindspore-lab/mindcv/blob/main/configs/densenet/densenet.png" width=800 />  
</p>
<p align="center">
  <em>Figure 1. Architecture of DenseNet</em>
</p>

## Results

<div align="center">

|    Model     | Context  | Top-1 (%) | Top-5 (%) | Params (M) |                                              Recipe                                                 |                                              Download                                            | 
|--------------|----------|-----------|-----------|------------|-----------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| densenet_121 | D910x8-G | 75.64     | 92.84     | 8.06       | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/densenet/densenet_121_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/densenet/densenet121-120_5004_Ascend.ckpt) |
| densenet_161 | D910x8-G | 79.09     | 94.66     | 28.90      | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/densenet/densenet_161_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/densenet/densenet161-120_5004_Ascend.ckpt) |
| densenet_169 | D910x8-G | 77.26     | 93.71     | 14.31      | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/densenet/densenet_169_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/densenet/densenet169-120_5004_Ascend.ckpt) |
| densenet_201 | D910x8-G | 78.14     | 94.08     | 20.24      | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/densenet/densenet_201_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/densenet/densenet201-120_5004_Ascend.ckpt) |

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
# distrubted training on multiple GPU/Ascend devices
mpirun -n 8 python train.py --config configs/densenet/densenet_121_ascend.yaml --data_dir /path/to/imagenet
```
  
Similarly, you can train the model on multiple GPU devices with the above `mpirun` command.

For detailed illustration of all hyper-parameters, please refer to [config.py](https://github.com/mindspore-lab/mindcv/blob/main/config.py).

**Note:**  As the global batch size  (batch_size x num_devices) is an important hyper-parameter, it is recommended to keep the global batch size unchanged for reproduction or adjust the learning rate linearly to a new global batch size.

* Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on a CPU/GPU/Ascend device
python train.py --config configs/densenet/densenet_121_ascend.yaml --data_dir /path/to/dataset --distribute False
```

### Validation

To validate the accuracy of the trained model, you can use `validate.py` and parse the checkpoint path with `--ckpt_path`.

```
python validate.py -c configs/densenet/densenet_121_ascend.yaml --data_dir /path/to/imagenet --ckpt_path /path/to/ckpt
```

### Deployment

Please refer to the [deployment tutorial](https://github.com/mindspore-lab/mindcv/blob/main/tutorials/deployment.md) in MindCV.
