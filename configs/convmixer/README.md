
# ConvMixer
> [Patches Are All You Need?](https://arxiv.org/pdf/2201.09792.pdf)

## Introduction

Although convolutional networks have been the dominant architecture for vision
tasks for many years, recent experiments have shown that Transformer-based models, most notably the Vision Transformer (ViT), may exceed their performance in
some settings. However, due to the quadratic runtime of the self-attention layers
in Transformers, ViTs require the use of patch embeddings, which group together
small regions of the image into single input features, in order to be applied to
larger image sizes. This raises a question: Is the performance of ViTs due to the
inherently-more-powerful Transformer architecture, or is it at least partly due to
using patches as the input representation? In this paper, we present some evidence
for the latter: specifically, we propose the ConvMixer, an extremely simple model
that is similar in spirit to the ViT and the even-more-basic MLP-Mixer in that it
operates directly on patches as input, separates the mixing of spatial and channel
dimensions, and maintains equal size and resolution throughout the network. In
contrast, however, the ConvMixer uses only standard convolutions to achieve the
mixing steps. Despite its simplicity, we show that the ConvMixer outperforms the
ViT, MLP-Mixer, and some of their variants for similar parameter counts and data
set sizes, in addition to outperforming classical vision models such as the ResNet.


## Results

**Implementation and configs for training were taken and adjusted from [this repository](https://gitee.com/cvisionlab/models/tree/convmixer/release/research/cv/convmixer), which implements ConvMixer models in mindspore.**

Our reproduced model performance on ImageNet-1K is reported as follows.

<div align="center">

| Model    | Context  | Top-1 (%) | Top-5 (%) | Params (M) | Recipe                                                                                        | Download                                                                               |
|----------|----------|-----------|-----------|------------|-----------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------|
| convmixer_768_32 | Converted from PyTorch | 79.68 | 94.92 | - | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/convmixer/convmixer_768_32.yaml) | [weights](https://storage.googleapis.com/huawei-mindspore-hk/ConvMixer/Converted/convmixer_768_32.ckpt) |
| convmixer_768_32 | 8xRTX3090 | 73.05 | 90.53 | - | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/convmixer/convmixer_768_32.yaml) | [weights](https://storage.googleapis.com/huawei-mindspore-hk/ConvMixer/convmixer_768_32_trained.ckpt) |
| convmixer_1024_20 | Converted from PyTorch | 76.68 | 93.3 | - | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/convmixer/convmixer_1024_20.yaml) | [weights](https://storage.googleapis.com/huawei-mindspore-hk/ConvMixer/Converted/convmixer_1024_20.ckpt) |
| convmixer_1536_20 | Converted from PyTorch | 80.98 | 95.51 | - | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/convmixer/convmixer_1536_20.yaml) | [weights](https://storage.googleapis.com/huawei-mindspore-hk/ConvMixer/Converted/convmixer_1536_20.ckpt) |


</div>

#### Notes

- Context: The weights in the table were taken from [official repository](https://github.com/locuslab/convmixer) and converted to mindspore format
- Top-1 and Top-5: Accuracy reported on the validation set of ImageNet-1K.

## Quick Start

### Preparation

#### Installation
Please refer to the [installation instruction](https://github.com/mindspore-ecosystem/mindcv#installation) in MindCV.

#### Dataset Preparation
Please download the [ImageNet-1K](https://www.image-net.org/challenges/LSVRC/2012/index.php) dataset for model training and validation.

### Training

* Distributed Training


```shell
# distrubted training on multiple GPU/Ascend devices
mpirun -n 8 python train.py --config configs/convmixer/convmixer_768_32.yaml --data_dir /path/to/imagenet --distributed True
```

> If the script is executed by the root user, the `--allow-run-as-root` parameter must be added to `mpirun`.

Similarly, you can train the model on multiple GPU devices with the above `mpirun` command.

For detailed illustration of all hyper-parameters, please refer to [config.py](https://github.com/mindspore-lab/mindcv/blob/main/config.py).

**Note:**  As the global batch size  (batch_size x num_devices) is an important hyper-parameter, it is recommended to keep the global batch size unchanged for reproduction or adjust the learning rate linearly to a new global batch size.

* Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on a CPU/GPU/Ascend device
python train.py --config configs/convmixer/convmixer_768_32.yaml --data_dir /path/to/dataset --distribute False
```

### Validation

To validate the accuracy of the trained model, you can use `validate.py` and parse the checkpoint path with `--ckpt_path`.

```shell
python validate.py -c configs/convmixer/convmixer_768_32.yaml --data_dir /path/to/imagenet --ckpt_path /path/to/ckpt
```

### Deployment

Please refer to the [deployment tutorial](https://github.com/mindspore-lab/mindcv/blob/main/tutorials/deployment.md) in MindCV.

## References

Paper - https://arxiv.org/pdf/2201.09792.pdf

Official repo - https://github.com/locuslab/convmixer

Mindspore implementation - https://gitee.com/cvisionlab/models/tree/convmixer/release/research/cv/convmixer
