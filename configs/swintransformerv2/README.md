# Swin Transformer V2
> [Swin Transformer V2: Scaling Up Capacity and Resolution](https://arxiv.org/abs/2111.09883)

## Introduction

This paper aims to explore large-scale models in computer vision. The authors tackle three major issues in training and
application of large vision models, including training instability, resolution gaps between pre-training and
fine-tuning, and hunger on labelled data. Three main techniques are proposed: 1) a residual-post-norm method combined
with cosine attention to improve training stability; 2) A log-spaced continuous position bias method to effectively
transfer models pre-trained using low-resolution images to downstream tasks with high-resolution inputs; 3) A
self-supervised pre-training method, SimMIM, to reduce the needs of vast labeled images. This model set new performance
records on 4 representative vision tasks, including ImageNet-V2 image classification, COCO object detection, ADE20K
semantic segmentation, and Kinetics-400 video action classification.[[1](#references)]

<p align="center">
  <img src="https://github.com/mindspore-lab/mindcv/assets/53842165/6ee39666-2852-408b-a31c-11cbdd85ac11" width=400 />
</p>
<p align="center">
  <em>Figure 1. Architecture of Swin Transformer V2 [<a href="#references">1</a>] </em>
</p>

## Results

Our reproduced model performance on ImageNet-1K is reported as follows.

<div align="center">

| Model                | Context  | Top-1 (%) | Top-5 (%) | Params (M) | Recipe                                                                                                              | Download                                                                                          |
|----------------------|----------|-----------|-----------|------------|---------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| swinv2_tiny_window8 | D910x8-G | 81.42     | 95.43     | 28.78      | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/swintransformerv2/swinv2_tiny_window8_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/swinv2/swinv2_tiny_window8-3ef8b787.ckpt) |

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
mpirun -n 8 python train.py --config configs/swintransformerv2/swinv2_tiny_window8_ascend.yaml --data_dir /path/to/imagenet
```

> If the script is executed by the root user, the `--allow-run-as-root` parameter must be added to `mpirun`.

Similarly, you can train the model on multiple GPU devices with the above `mpirun` command.

For detailed illustration of all hyper-parameters, please refer to [config.py](https://github.com/mindspore-lab/mindcv/blob/main/config.py).

**Note:**  As the global batch size  (batch_size x num_devices) is an important hyper-parameter, it is recommended to keep the global batch size unchanged for reproduction or adjust the learning rate linearly to a new global batch size.

* Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on a CPU/GPU/Ascend device
python train.py --config configs/swintransformerv2/swinv2_tiny_window8_ascend.yaml --data_dir /path/to/dataset --distribute False
```

### Validation

To validate the accuracy of the trained model, you can use `validate.py` and parse the checkpoint path with `--ckpt_path`.

```shell
python validate.py -c configs/swintransformerv2/swinv2_tiny_window8_ascend.yaml --data_dir /path/to/imagenet --ckpt_path /path/to/ckpt
```

### Deployment

Please refer to the [deployment tutorial](https://mindspore-lab.github.io/mindcv/tutorials/deployment/) in MindCV.

## References

[1] Liu Z, Hu H, Lin Y, et al. Swin transformer v2: Scaling up capacity and resolution[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022: 12009-12019.
