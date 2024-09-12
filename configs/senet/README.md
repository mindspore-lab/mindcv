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

performance tested on ascend 910*(8p) with graph mode

<div align="center">

|   Model    | Top-1 (%) | Top-5 (%) | ms/step | Params (M) | Batch Size | Recipe                                                                                         | Download                                                                                              |
| :--------: | :-------: | :-------: | :-----: | :--------: | ---------- | ---------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| seresnet18 |   72.05   |   90.59   |  48.72  |   11.80    | 64         | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/senet/seresnet18_ascend.yaml) | [weights](https://download-mindspore.osinfra.cn/toolkits/mindcv/senet/seresnet18-7b971c78-910v2.ckpt) |

</div>

performance tested on ascend 910(8p) with graph mode

<div align="center">

|   Model    | Top-1 (%) | Top-5 (%) | Params (M) | Batch Size | Recipe                                                                                         | Download                                                                                |
|:----------:|:---------:|:---------:|:----------:|------------|------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|
| seresnet18 |   71.81   |   90.49   |   11.80    | 64         | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/senet/seresnet18_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/senet/seresnet18-7880643b.ckpt) |

</div>

#### Notes

- Top-1 and Top-5: Accuracy reported on the validation set of ImageNet-1K.

## Quick Start

### Preparation

#### Installation

Please refer to the [installation instruction](https://github.com/mindspore-ecosystem/mindcv#installation) in MindCV.

#### Dataset Preparation

Please download the [ImageNet-1K](https://www.image-net.org/challenges/LSVRC/2012/index.php) dataset for model training
and validation.

### Training

* Distributed Training

It is easy to reproduce the reported results with the pre-defined training recipe. For distributed training on multiple
Ascend 910 devices, please run

```shell
# distributed training on multiple GPU/Ascend devices
msrun --bind_core=True --worker_num 8 python train.py --config configs/senet/seresnet50_ascend.yaml --data_dir /path/to/imagenet
```



Similarly, you can train the model on multiple GPU devices with the above `msrun` command.

For detailed illustration of all hyper-parameters, please refer
to [config.py](https://github.com/mindspore-lab/mindcv/blob/main/config.py).

**Note:**  As the global batch size  (batch_size x num_devices) is an important hyper-parameter, it is recommended to
keep the global batch size unchanged for reproduction or adjust the learning rate linearly to a new global batch size.

* Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on a CPU/GPU/Ascend device
python train.py --config configs/senet/seresnet50_ascend.yaml --data_dir /path/to/dataset --distribute False
```

### Validation

To validate the accuracy of the trained model, you can use `validate.py` and parse the checkpoint path
with `--ckpt_path`.

```shell
python validate.py -c configs/senet/seresnet50_ascend.yaml --data_dir /path/to/imagenet --ckpt_path /path/to/ckpt
```

### Deployment

Please refer to the [deployment tutorial](https://mindspore-lab.github.io/mindcv/tutorials/deployment/) in MindCV.

## References

[1] Hu J, Shen L, Sun G. Squeeze-and-excitation networks[C]//Proceedings of the IEEE conference on computer vision and
pattern recognition. 2018: 7132-7141.
