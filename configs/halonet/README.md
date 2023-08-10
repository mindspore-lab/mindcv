# HaloNet

> [Scaling Local Self-Attention for Parameter Efficient Visual Backbones](https://arxiv.org/abs/2103.12731)

## Introduction

Researchers from Google Research and UC Berkeley have developed a new model of self-attention that can outperform standard baseline models and even high-performance convolutional models.[[1](#references)]

Blocked Self-Attention：The whole input image is divided into multiple blocks and self-attention is applied to each block.However, if only the information inside the block is considered each time, it will inevitably lead to the loss of information.Therefore, before calculating the SA, a haloing operation is performed on each block, i.e., outside of each block, the information of the original image is used to padding a circle, so that the sensory field of each block can be appropriately larger and focus on more information.

<p align="center">
  <img src="https://github-production-user-asset-6210df.s3.amazonaws.com/50255437/257577202-3ac43b82-785a-42c5-9b6c-ca58b0fa7ab8.png" width=800 />
</p>
<p align="center">
  <em>Figure 1. Architecture of Blocked Self-Attention [<a href="#references">1</a>] </em>
</p>

Down Sampling：In order to reduce the amount of computation, each block is sampled separately, and then attentions are performed on this sampled information to reach the effect of down sampling.

<p align="center">
  <img src="https://github-production-user-asset-6210df.s3.amazonaws.com/50255437/257578183-fe45c2c2-5006-492b-b30a-5b049a0e2531.png" width=800 />
</p>
<p align="center">
  <em>Figure 2. Architecture of Down Sampling [<a href="#references">1</a>] </em>
</p>


## Results

Our reproduced model performance on ImageNet-1K is reported as follows.

<div align="center">

| Model       | Context  | Top-1 (%) | Top-5 (%) | Params (M) | Recipe                                                       | Download                                                     |
| ----------- | -------- | --------- | --------- | ---------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| halonet_50t | D910X8-G | 79.53     | 94.79     | 22.79      | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/halonet/halonet_50t_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/halonet/halonet_50t-533da6be.ckpt) |

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
mpirun -n 8 python train.py --config configs/halonet/halonet_50t_ascend.yaml  --data_dir /path/to/imagenet
```

> If the script is executed by the root user, the `--allow-run-as-root` parameter must be added to `mpirun`.

Similarly, you can train the model on multiple GPU devices with the above `mpirun` command.

For detailed illustration of all hyper-parameters, please refer to [config.py](https://github.com/mindspore-lab/mindcv/blob/main/config.py).

**Note:**  As the global batch size  (batch_size x num_devices) is an important hyper-parameter, it is recommended to keep the global batch size unchanged for reproduction or adjust the learning rate linearly to a new global batch size.

* Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on a CPU/GPU/Ascend device
python train.py --config configs/halonet/halonet_50t_ascend.yaml  --data_dir /path/to/dataset --distribute False
```

### Validation

To validate the accuracy of the trained model, you can use `validate.py` and parse the checkpoint path with `--ckpt_path`.

```shell
python validate.py -c configs/halonet/halonet_50t_ascend.yaml  --data_dir /path/to/imagenet --ckpt_path /path/to/ckpt
```

### Deployment

Please refer to the [deployment tutorial](https://mindspore-lab.github.io/mindcv/tutorials/deployment/) in MindCV.

## References

[1] Vaswani A, Ramachandran P, Srinivas A, et al. Scaling local self-attention for parameter efficient visual backbones[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021: 12894-12904.
