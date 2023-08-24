# ResNetV2

> [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027)

## Introduction

Author analyzes the propagation formulations behind the residual building blocks, which suggest that the forward and backward signals can be directly propagated from one block
to any other block, when using identity mappings as the skip connections and after-addition activation.

<p align="center">
  <img src="https://user-images.githubusercontent.com/52945530/224595993-ba8617da-e55d-4d19-a487-3340026393c9.png" width=300 height=400 />
</p>
<p align="center">
  <em>Figure 1. Architecture of ResNetV2 [<a href="#references">1</a>] </em>
</p>

## Results

Our reproduced model performance on ImageNet-1K is reported as follows.

<div align="center">

| Model        | Context   | Top-1 (%) | Top-5 (%) |  Params (M) | Recipe  | Download                                                                           |
|--------------|-----------|-----------|-----------|-------|-------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------|
| resnetv2_50  | D910x8-G | 76.90     | 93.37     | 25.60 | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/resnetv2/resnetv2_50_ascend.yaml)  | [weights](https://download.mindspore.cn/toolkits/mindcv/resnetv2/resnetv2_50-3c2f143b.ckpt)  |
| resnetv2_101 | D910x8-G | 78.48     | 94.23     | 44.55 | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/resnetv2/resnetv2_101_ascend.yaml)  | [weights](https://download.mindspore.cn/toolkits/mindcv/resnetv2/resnetv2_101-5d4c49a1.ckpt)  |

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
# distributed training on multiple GPU/Ascend devices
mpirun -n 8 python train.py --config configs/resnetv2/resnetv2_50_ascend.yaml --data_dir /path/to/imagenet
```

> If the script is executed by the root user, the `--allow-run-as-root` parameter must be added to `mpirun`.

Similarly, you can train the model on multiple GPU devices with the above `mpirun` command.

For detailed illustration of all hyper-parameters, please refer to [config.py](https://github.com/mindspore-lab/mindcv/blob/main/config.py).

**Note:**  As the global batch size  (batch_size x num_devices) is an important hyper-parameter, it is recommended to keep the global batch size unchanged for reproduction or adjust the learning rate linearly to a new global batch size.

* Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on a CPU/GPU/Ascend device
python train.py --config configs/resnetv2/resnetv2_50_ascend.yaml --data_dir /path/to/dataset --distribute False
```

### Validation

To validate the accuracy of the trained model, you can use `validate.py` and parse the checkpoint path with `--ckpt_path`.

```shell
python validate.py -c configs/resnetv2/resnetv2_50_ascend.yaml --data_dir /path/to/imagenet --ckpt_path /path/to/ckpt
```

### Deployment

To deploy online inference services with the trained model efficiently, please refer to the [deployment tutorial](https://mindspore-lab.github.io/mindcv/tutorials/deployment/).

## References

[1] He K, Zhang X, Ren S, et al. Identity mappings in deep residual networks[C]//Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11–14, 2016, Proceedings, Part IV 14. Springer International Publishing, 2016: 630-645.
