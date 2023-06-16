# MobileViT
> [MobileViT：Light-weight, General-purpose, and Mobile-friendly Vision Transformer](https://arxiv.org/pdf/2110.02178.pdf)

## Introduction

MobileViT, a light-weight and general-purpose vision transformer for mobile devices. MobileViT presents a different perspective for the global processing of information with transformers, i.e., transformers as convolutions. MobileViT significantly outperforms CNN- and ViT-based networks across different tasks and datasets. On the ImageNet-1k dataset, MobileViT achieves top-1 accuracy of 78.4% with about 6 million parameters, which is 3.2% and 6.2% more accurate than MobileNetv3 (CNN-based) and DeIT (ViT-based) for a similar number of parameters. On the MS-COCO object detection task, MobileViT is 5.7% more accurate than MobileNetv3 for a similar number of parameters.

<p align="center">
  <img src="https://user-images.githubusercontent.com/64628185/229476902-1b97496a-4a38-40ca-9e50-a88c52defcbb.png" width=800 />
</p>
<p align="center">
  <em>Figure 1. Architecture of MobileViT [<a href="#references">1</a>] </em>
</p>

## Results

Our reproduced model performance on ImageNet-1K is reported as follows.

<div align="center">

| Model       | Context  | Top-1 (%) | Top-5 (%) | Params (M) | Recipe                                                                                  | Download                                                                              |
|-------------|----------|-----------|-----------|------------|-----------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------|
| mobilevit_xx_small | D910x8-G | 68.91 | 88.91 | 1.27 | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/mobilevit/mobilevit_xx_small_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/mobilevit/mobilevit_xx_small-af9da8a0.ckpt) |
| mobilevit_x_small | D910x8-G | 74.99 | 92.32 | 2.32 | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/mobilevit/mobilevit_x_small_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/mobilevit/mobilevit_x_small-673fc6f2.ckpt) |
| mobilevit_small | D910x8-G | 78.47 | 94.18 | 5.59 | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/mobilevit/mobilevit_small_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/mobilevit/mobilevit_small-caf79638.ckpt) |

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
mpirun -n 8 python train.py --config configs/mobilevit/mobilevit_xx_small_ascend.yaml --data_dir /path/to/imagenet
```
> If the script is executed by the root user, the `--allow-run-as-root` parameter must be added to `mpirun`.

Similarly, you can train the model on multiple GPU devices with the above `mpirun` command.

For detailed illustration of all hyper-parameters, please refer to [config.py](https://github.com/mindspore-lab/mindcv/blob/main/config.py).

**Note:**  As the global batch size  (batch_size x num_devices) is an important hyper-parameter, it is recommended to keep the global batch size unchanged for reproduction or adjust the learning rate linearly to a new global batch size.

* Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on a CPU/GPU/Ascend device
python train.py --config configs/mobilevit/mobilevit_xx_small_ascend.yaml --data_dir /path/to/dataset --distribute False
```

### Validation

To validate the accuracy of the trained model, you can use `validate.py` and parse the checkpoint path with `--ckpt_path`.

```
python validate.py -c configs/mobilevit/mobilevit_xx_small_ascend.yaml --data_dir /path/to/imagenet --ckpt_path /path/to/ckpt
```

### Deployment

Please refer to the [deployment tutorial](https://mindspore-lab.github.io/mindcv/tutorials/deployment/) in MindCV.
