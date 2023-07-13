# ResNeSt
> [ResNeSt: Split-Attention Networks](https://arxiv.org/abs/2004.08955)

## Introduction

In this paper, the authors present a modularized architecture, which applies the channel-wise attention on different
network branches to leverage their success in capturing cross-feature interactions and learning diverse representations.
The network design results in a simple and unified computation block, which can be parameterized using only a few
variables. As a result, ResNeSt outperforms EfficientNet in accuracy and latency trade-off on image
classification.[[1](#references)]

<p align="center">
  <img src="https://user-images.githubusercontent.com/53842165/223672204-8ac59c6c-cd8a-45c2-945f-7e556c383056.jpg" width=500 />
</p>
<p align="center">
  <em>Figure 1. Architecture of ResNeSt [<a href="#references">1</a>] </em>
</p>

## Results

Our reproduced model performance on ImageNet-1K is reported as follows.

<div align="center">

| Model      | Context  | Top-1 (%) | Top-5 (%) | Params (M) | Recipe                                                                                           | Download                                                                                  |
|------------|----------|-----------|-----------|------------|--------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| resnest50  | D910x8-G | 80.81     | 95.16     | 27.55      | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/resnest/resnest50_ascend.yaml)  | [weights](https://download.mindspore.cn/toolkits/mindcv/resnest/resnest50-f2e7fc9c.ckpt)  |
| resnest101 | D910x8-G | 82.90     | 96.12     | 48.41      | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/resnest/resnest101_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/resnest/resnest101-7cc5c258.ckpt) |

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
mpirun -n 8 python train.py --config configs/resnest/resnest50_ascend.yaml --data_dir /path/to/imagenet
```

> If the script is executed by the root user, the `--allow-run-as-root` parameter must be added to `mpirun`.

Similarly, you can train the model on multiple GPU devices with the above `mpirun` command.

For detailed illustration of all hyper-parameters, please refer to [config.py](https://github.com/mindspore-lab/mindcv/blob/main/config.py).

**Note:**  As the global batch size  (batch_size x num_devices) is an important hyper-parameter, it is recommended to keep the global batch size unchanged for reproduction or adjust the learning rate linearly to a new global batch size.

* Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on a CPU/GPU/Ascend device
python train.py --config configs/resnest/resnest50_ascend.yaml --data_dir /path/to/dataset --distribute False
```

### Validation

To validate the accuracy of the trained model, you can use `validate.py` and parse the checkpoint path with `--ckpt_path`.

```shell
python validate.py -c configs/resnest/resnest50_ascend.yaml --data_dir /path/to/imagenet --ckpt_path /path/to/ckpt
```

### Deployment

Please refer to the [deployment tutorial](https://mindspore-lab.github.io/mindcv/tutorials/deployment/) in MindCV.

## References

[1] Zhang H, Wu C, Zhang Z, et al. Resnest: Split-attention networks[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022: 2736-2746.
