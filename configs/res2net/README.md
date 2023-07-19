# Res2Net

> [Res2Net: A New Multi-scale Backbone Architecture](https://arxiv.org/abs/1904.01169)

## Introduction

Res2Net is a novel building block for CNNs proposed by constructing hierarchical residual-like connections within one single residual block. The Res2Net represents multi-scale features at a granular level and increases the range of receptive fields for each network layer. Res2Net block can be plugged into the state-of-the-art backbone CNN models, e.g., ResNet, ResNeXt, and DLA. Ablation studies and experimental results on representative computer vision tasks, i.e., object detection, class activation mapping, and salient object detection, verify the superiority of the Res2Net over the state-of-the-art baseline methods such as ResNet-50, DLA-60 and etc.

<p align="center">
  <img src="https://user-images.githubusercontent.com/121591093/210049799-ee3971d5-fad9-41d2-a8cd-ef64aa9d4724.png" width=800 />
</p>
<p align="center">
  <em>Figure 1. Architecture of Res2Net [<a href="#references">1</a>] </em>
</p>

## Results

Our reproduced model performance on ImageNet-1K is reported as follows.

<div align="center">

| Model          | Context   | Top-1 (%) | Top-5 (%)  |  Params (M)    | Recipe                                                                                                |  Download |
|----------------|-----------|-----------|-------|------------|-------------------------------------------------------------------------------------------------------|---|
| res2net50      | D910x8-G | 79.35     | 94.64     | 25.76     | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/res2net/res2net_50_ascend.yaml)      | [weights](https://download.mindspore.cn/toolkits/mindcv/res2net/res2net50-f42cf71b.ckpt)  |
| res2net101     | D910x8-G | 79.56     | 94.70     | 45.33     | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/res2net/res2net_101_ascend.yaml)     | [weights](https://download.mindspore.cn/toolkits/mindcv/res2net/res2net101-8ae60132.ckpt)  |
| res2net50_v1b  | D910x8-G | 80.32     | 95.09     | 25.77   | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/res2net/res2net_50_v1b_ascend.yaml)  | [weights](https://download.mindspore.cn/toolkits/mindcv/res2net/res2net50_v1b-99304e92.ckpt)  |
| res2net101_v1b | D910x8-G | 81.14     | 95.41     | 45.35 | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/res2net/res2net_101_v1b_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/res2net/res2net101_v1b-7e6db001.ckpt)  |

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
mpirun -n 8 python train.py --config configs/res2net/res2net_50_ascend.yaml --data_dir /path/to/imagenet
```

> If the script is executed by the root user, the `--allow-run-as-root` parameter must be added to `mpirun`.

Similarly, you can train the model on multiple GPU devices with the above `mpirun` command.

For detailed illustration of all hyper-parameters, please refer to [config.py](https://github.com/mindspore-lab/mindcv/blob/main/config.py).

**Note:**  As the global batch size  (batch_size x num_devices) is an important hyper-parameter, it is recommended to keep the global batch size unchanged for reproduction or adjust the learning rate linearly to a new global batch size.

* Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on a CPU/GPU/Ascend device
python train.py --config configs/res2net/res2net_50_ascend.yaml --data_dir /path/to/imagenet --distribute False
```

### Validation

To validate the accuracy of the trained model, you can use `validate.py` and parse the checkpoint path with `--ckpt_path`.

```shell
python validate.py -c configs/res2net/res2net_50_ascend.yaml --data_dir /path/to/imagenet --ckpt_path /path/to/ckpt
```

### Deployment

To deploy online inference services with the trained model efficiently, please refer to the [deployment tutorial](https://mindspore-lab.github.io/mindcv/tutorials/deployment/).

## References

[1] Gao S H, Cheng M M, Zhao K, et al. Res2net: A new multi-scale backbone architecture[J]. IEEE transactions on pattern analysis and machine intelligence, 2019, 43(2): 652-662.
