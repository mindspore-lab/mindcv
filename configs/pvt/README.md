# Pyramid Vision Transformer

> [Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions](https://arxiv.org/abs/2102.12122)

## Introduction

PVT is a general backbone network for dense prediction without convolution operation. PVT introduces a pyramid structure in Transformer to generate multi-scale feature maps for dense prediction tasks. PVT uses a gradual reduction strategy to control the size of the feature maps through the patch embedding layer, and proposes a spatial reduction attention (SRA) layer to replace the traditional multi head attention layer in the encoder, which greatly reduces the computing/memory overhead.[[1](#References)]

![PVT](https://user-images.githubusercontent.com/74176172/210046926-2322161b-a963-4603-b3cb-86ecdca41262.png)

## Results

Our reproduced model performance on ImageNet-1K is reported as follows.

<div align="center">

|   Model    | Context  | Top-1 (%) | Top-5 (%) | Params (M) | Recipe                                                                                       | Download                                                                         |
|:----------:|:--------:|:---------:|:---------:|:----------:|----------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
|  pvt_tiny  | D910x8-G |   74.81   |   92.18   |   13.23    | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/pvt/pvt_tiny_ascend.yaml)   | [weights](https://download.mindspore.cn/toolkits/mindcv/pvt/pvt_tiny-6abb953d.ckpt)   |
| pvt_small  | D910x8-G |   79.66   |   94.71   |   24.49    | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/pvt/pvt_small_ascend.yaml)  | [weights](https://download.mindspore.cn/toolkits/mindcv/pvt/pvt_small-213c2ed1.ckpt)  |
| pvt_medium | D910x8-G |   81.82   |   95.81   |   44.21    | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/pvt/pvt_medium_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/pvt/pvt_medium-469e6802.ckpt) |
| pvt_large  | D910x8-G |   81.75   |   95.70   |   61.36    | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/pvt/pvt_large_ascend.yaml)  | [weights](https://download.mindspore.cn/toolkits/mindcv/pvt/pvt_large-bb6895d7.ckpt)  |

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
mpirun -n 8 python train.py --config configs/pvt/pvt_tiny_ascend.yaml --data_dir /path/to/imagenet
```
> If the script is executed by the root user, the `--allow-run-as-root` parameter must be added to `mpirun`.
> If use Ascend 910 devices, need to open SATURATION_MODE via `export MS_ASCEND_CHECK_OVERFLOW_MODE="SATURATION_MODE"`

Similarly, you can train the model on multiple GPU devices with the above `mpirun` command.

For detailed illustration of all hyper-parameters, please refer to [config.py](https://github.com/mindspore-lab/mindcv/blob/main/config.py).

**Note:** As the global batch size (batch_size x num_devices) is an important hyper-parameter, it is recommended to keep the global batch size unchanged for reproduction or adjust the learning rate linearly to a new global batch size.

* Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on a CPU/GPU/Ascend device
python train.py --config configs/pvt/pvt_tiny_ascend.yaml --data_dir /path/to/imagenet --distribute False
```
> If use Ascend 910 devices, need to open SATURATION_MODE via `export MS_ASCEND_CHECK_OVERFLOW_MODE="SATURATION_MODE"`

### Validation

To validate the accuracy of the trained model, you can use `validate.py` and parse the checkpoint path with `--ckpt_path`.

```shell
python validate.py --model=pvt_tiny --data_dir /path/to/imagenet --ckpt_path /path/to/ckpt
```

### Deployment

To deploy online inference services with the trained model efficiently, please refer to the [deployment tutorial](https://mindspore-lab.github.io/mindcv/tutorials/deployment/).

## References

[1]. Wang W, Xie E, Li X, et al. Pyramid vision transformer: A versatile backbone for dense prediction without convolutions[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021: 568-578.
