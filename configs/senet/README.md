# SENet
> [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)

## Introduction

In this work, the authors focus instead on the channel relationship and propose a novel architectural unit, which the
authors term the "Squeeze-and-Excitation" (SE) block, that adaptively recalibrates channel-wise feature responses by
explicitly modelling interdependencies between channels. The results show that these blocks can be stacked together to
form SENet architectures that generalise extremely effectively across different datasets. The authors further
demonstrate that SE blocks bring significant improvements in performance for existing state-of-the-art CNNs at slight
additional computational cost.[[1](#references)]

<p align="center">
  <img src="https://user-images.githubusercontent.com/53842165/218919253-618d3d66-9b2a-4e27-b866-a21015cd9600.png" width=800 />
</p>
<p align="center">
  <em>Figure 1. Architecture of SENet [<a href="#references">1</a>] </em>
</p>

## Results

Our reproduced model performance on ImageNet-1K is reported as follows.

<div align="center">

| Model             | Context  | Top-1 (%) | Top-5 (%) | Params (M) | Recipe                                                                                                | Download                                                                                       |
|-------------------|----------|-----------|-----------|------------|-------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|
| seresnet18        | D910x8-G | 71.81     | 90.49     | 11.80      | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/senet/seresnet18_ascend.yaml)        | [weights](https://download.mindspore.cn/toolkits/mindcv/senet/seresnet18-7880643b.ckpt)        |
| seresnet34        | D910x8-G | 75.38     | 92.50     | 21.98      | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/senet/seresnet34_ascend.yaml)        | [weights](https://download.mindspore.cn/toolkits/mindcv/senet/seresnet34-8179d3c9.ckpt)        |
| seresnet50        | D910x8-G | 78.32     | 94.07     | 28.14      | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/senet/seresnet50_ascend.yaml)        | [weights](https://download.mindspore.cn/toolkits/mindcv/senet/seresnet50-ff9cd214.ckpt)        |
| seresnext26_32x4d | D910x8-G | 77.17     | 93.42     | 16.83      | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/senet/seresnext26_32x4d_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/senet/seresnext26_32x4d-5361f5b6.ckpt) |
| seresnext50_32x4d | D910x8-G | 78.71     | 94.36     | 27.63      | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/senet/seresnext50_32x4d_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/senet/seresnext50_32x4d-fdc35aca.ckpt) |

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
mpirun -n 8 python train.py --config configs/senet/seresnet50_ascend.yaml --data_dir /path/to/imagenet
```

> If the script is executed by the root user, the `--allow-run-as-root` parameter must be added to `mpirun`.

Similarly, you can train the model on multiple GPU devices with the above `mpirun` command.

For detailed illustration of all hyper-parameters, please refer to [config.py](https://github.com/mindspore-lab/mindcv/blob/main/config.py).

**Note:**  As the global batch size  (batch_size x num_devices) is an important hyper-parameter, it is recommended to keep the global batch size unchanged for reproduction or adjust the learning rate linearly to a new global batch size.

* Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on a CPU/GPU/Ascend device
python train.py --config configs/senet/seresnet50_ascend.yaml --data_dir /path/to/dataset --distribute False
```

### Validation

To validate the accuracy of the trained model, you can use `validate.py` and parse the checkpoint path with `--ckpt_path`.

```shell
python validate.py -c configs/senet/seresnet50_ascend.yaml --data_dir /path/to/imagenet --ckpt_path /path/to/ckpt
```

### Deployment

Please refer to the [deployment tutorial](https://mindspore-lab.github.io/mindcv/tutorials/deployment/) in MindCV.

## References

[1] Hu J, Shen L, Sun G. Squeeze-and-excitation networks[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2018: 7132-7141.
