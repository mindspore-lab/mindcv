# CrossViT
> [CrossViT: Cross-Attention Multi-Scale Vision Transformer for Image Classification](https://arxiv.org/abs/2103.14899)

## Introduction

CrossViT is a type of vision transformer that uses a dual-branch architecture to extract multi-scale feature representations for image classification. The architecture combines image patches (i.e. tokens in a transformer) of different sizes to produce stronger visual features for image classification. It processes small and large patch tokens with two separate branches of different computational complexities and these tokens are fused together multiple times to complement each other.

Fusion is achieved by an efficient cross-attention module, in which each transformer branch creates a non-patch token as an agent to exchange information with the other branch by attention. This allows for linear-time generation of the attention map in fusion instead of quadratic time otherwise.[[1](#references)]

<p align="center">
  <img src="https://user-images.githubusercontent.com/52945530/223635248-5871596d-43f2-44ee-b8be-1e7927ade243.jpg" width=400 />
</p>
<p align="center">
  <em>Figure 1. Architecture of CrossViT [<a href="#references">1</a>] </em>
</p>


## Results

Our reproduced model performance on ImageNet-1K is reported as follows.

<div align="center">

| Model       | Context  | Top-1 (%) | Top-5 (%) | Params (M) | Recipe                                                                                  | Download                                                                              |
|-------------|----------|-----------|-----------|------------|-----------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------|
| crossvit_9 | D910x8-G | 73.56     | 91.79     | 8.55      | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/crossvit/crossvit_9_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/crossvit/crossvit_9-e74c8e18.ckpt)     |
| crossvit_15 | D910x8-G | 81.08     | 95.33     | 27.27      | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/crossvit/crossvit_15_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/crossvit/crossvit_15-eaa43c02.ckpt)     |
| crossvit_18 | D910x8-G | 81.93     | 95.75     | 43.27      | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/crossvit/crossvit_18_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/crossvit/crossvit_18-ca0a2e43.ckpt) |


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
mpirun -n 8 python train.py --config configs/crossvit/crossvit_15_ascend.yaml --data_dir /path/to/imagenet
```
> If the script is executed by the root user, the `--allow-run-as-root` parameter must be added to `mpirun`.

Similarly, you can train the model on multiple GPU devices with the above `mpirun` command.

For detailed illustration of all hyper-parameters, please refer to [config.py](https://github.com/mindspore-lab/mindcv/blob/main/config.py).

**Note:**  As the global batch size  (batch_size x num_devices) is an important hyper-parameter, it is recommended to keep the global batch size unchanged for reproduction or adjust the learning rate linearly to a new global batch size.

* Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on a CPU/GPU/Ascend device
python train.py --config configs/crossvit/crossvit_15_ascend.yaml --data_dir /path/to/dataset --distribute False
```

### Validation

To validate the accuracy of the trained model, you can use `validate.py` and parse the checkpoint path with `--ckpt_path`.

```
python validate.py -c configs/crossvit/crossvit_15_ascend.yaml --data_dir /path/to/imagenet --ckpt_path /path/to/ckpt
```

### Deployment

Please refer to the [deployment tutorial](https://mindspore-lab.github.io/mindcv/tutorials/deployment/) in MindCV.

## References

<!--- Guideline: Citation format should follow GB/T 7714. -->
[1] Chun-Fu Chen, Quanfu Fan, Rameswar Panda. CrossViT: Cross-Attention Multi-Scale Vision Transformer for Image Classification
