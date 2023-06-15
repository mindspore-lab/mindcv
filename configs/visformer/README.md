# Visformer
>  [Visformer: The Vision-friendly Transformer](https://arxiv.org/abs/2104.12533)

## Introduction

Visformer, or Vision-friendly Transformer, is an architecture that combines Transformer-based architectural features with those from convolutional neural network architectures. Visformer adopts the stage-wise design for higher base performance. But self-attentions are only utilized in the last two stages, considering that self-attention in the high-resolution stage is relatively inefficient even when the FLOPs are balanced. Visformer employs bottleneck blocks in the first stage and utilizes group 3 × 3 convolutions in bottleneck blocks inspired by ResNeXt. It also introduces BatchNorm to patch embedding modules as in CNNs. [[2](#references)]

<p align="center">
  <img src="https://user-images.githubusercontent.com/8156835/210043162-8b861b87-73ba-4376-aae9-534496896c37.png" width=800 />
</p>
<p align="center">
  <em>Figure 1. Network Configuration of Visformer  [<a href="https://arxiv.org/abs/2104.12533">1</a>] </em>
</p>

## Results

## ImageNet-1k

Our reproduced model performance on ImageNet-1K is reported as follows.

<div align="center">

| Model              | Context  | Top-1 (%) | Top-5 (%) | Params (M) | Recipe                                                                                                     | Download                                                                                   |
|--------------------|----------|-----------|-----------|------------|------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|
| visformer_tiny     | D910x8-G | 78.28     | 94.15     | 10.33      | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/visformer/visformer_tiny_ascend.yaml)     | [weights](https://download.mindspore.cn/toolkits/mindcv/visformer/visformer_tiny-daee0322.ckpt)     |
| visformer_tiny_v2  | D910x8-G | 78.82     | 94.41     | 9.38       | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/visformer/visformer_tiny_v2_ascend.yaml)  | [weights](https://download.mindspore.cn/toolkits/mindcv/visformer/visformer_tiny_v2-6711a758.ckpt)  |
| visformer_small    | D910x8-G | 81.76     | 95.88     | 40.25      | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/visformer/visformer_small_ascend.yaml)    | [weights](https://download.mindspore.cn/toolkits/mindcv/visformer/visformer_small-6c83b6db.ckpt)    |
| visformer_small_v2 | D910x8-G | 82.17     | 95.90     | 23.52      | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/visformer/visformer_small_v2_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/visformer/visformer_small_v2-63674ade.ckpt) |

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
mpirun -n 8 python train.py --config configs/visformer/visformer_tiny_ascend.yaml --data_dir /path/to/imagenet
```

Similarly, you can train the model on multiple GPU devices with the above `mpirun` command.

For detailed illustration of all hyper-parameters, please refer to [config.py](https://github.com/mindspore-lab/mindcv/blob/main/config.py).

**Note:**  As the global batch size  (batch_size x num_devices) is an important hyper-parameter, it is recommended to keep the global batch size unchanged for reproduction or adjust the learning rate linearly to a new global batch size.

* Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on a CPU/GPU/Ascend device
python train.py --config configs/visformer/visformer_tiny_ascend.yaml --data_dir /path/to/dataset --distribute False
```

### Validation

To validate the accuracy of the trained model, you can use `validate.py` and parse the checkpoint path with `--ckpt_path`.

```
python validate.py -c configs/visformer/visformer_tiny_ascend.yaml --data_dir /path/to/imagenet --ckpt_path /path/to/ckpt
```

### Deployment

To deploy online inference services with the trained model efficiently, please refer to the [deployment tutorial](https://mindspore-lab.github.io/mindcv/tutorials/deployment/).

## References
[1] Chen Z, Xie L, Niu J, et al. Visformer: The vision-friendly transformer. Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021: 589-598.

[2] Visformer, https://paperswithcode.com/method/visformer
