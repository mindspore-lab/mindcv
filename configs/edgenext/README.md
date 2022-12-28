# EdgeNeXt

> [EdgeNeXt: Efficiently Amalgamated CNN-Transformer Architecture for Mobile Vision Applications](https://arxiv.org/abs/2206.10589)

## Introduction


In the pursuit of achieving ever-increasing accuracy, large and complex neural networks are usually developed. Such models demand high computational resources and therefore cannot be deployed on edge devices. It is of great interest to build resource-efficient general-purpose networks due to their usefulness in several application areas. This paper strives to effectively combine the strengths of both CNN and Transformer models and propose a new efficient hybrid architecture EdgeNeXt. EdgeNeXt introduces split depth-wise transpose attention (SDTA) encoder that splits input tensors into multiple channel groups and utilizes depth-wise convolution along with self-attention across channel dimensions to implicitly increase the receptive field and encode multi-scale features.

<p align="center">
  <img src="https://github.com/mindspore-lab/mindcv/blob/main/configs/edgenext/edgenext.png" width=800 />  
</p>
<p align="center">
  <em>Figure 1. Architecture of EdgeNeXt</em>
</p>

## Results

<div align="center">

| Model          | Context  |  Top-1 (%) | Top-5 (%) |  Params (M) |                                                 Recipe                                                |                                      Download                                       | 
|----------------|----------|------------|-----------|-------------|-------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| edgenext_small | D910x8-G | 79.15      |  94.39    | 5.59        | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/edgenext/edgenext_small_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/edgenext/edgenext_small.ckpt) |

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
# distrubted training on multiple GPU/Ascend devices
mpirun -n 8 python train.py --config configs/edgenext/edgenext_small_ascend.yaml --data_dir /path/to/imagenet
```
  
Similarly, you can train the model on multiple GPU devices with the above `mpirun` command.

For detailed illustration of all hyper-parameters, please refer to [config.py](https://github.com/mindspore-lab/mindcv/blob/main/config.py).

**Note:**  As the global batch size  (batch_size x num_devices) is an important hyper-parameter, it is recommended to keep the global batch size unchanged for reproduction or adjust the learning rate linearly to a new global batch size.

* Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on a CPU/GPU/Ascend device
python train.py --config configs/edgenext/edgenext_small_ascend.yaml --data_dir /path/to/dataset --distribute False
```

### Validation

To validate the accuracy of the trained model, you can use `validate.py` and parse the checkpoint path with `--ckpt_path`.

```
python validate.py -c configs/edgenext/edgenext_small_ascend.yaml --data_dir /path/to/imagenet --ckpt_path /path/to/ckpt
```

### Deployment

Please refer to the [deployment tutorial](https://github.com/mindspore-lab/mindcv/blob/main/tutorials/deployment.md) in MindCV.
