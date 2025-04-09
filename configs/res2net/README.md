# Res2Net

> [Res2Net: A New Multi-scale Backbone Architecture](https://arxiv.org/abs/1904.01169)



## Introduction

Res2Net is a novel building block for CNNs proposed by constructing hierarchical residual-like connections within one
single residual block. The Res2Net represents multi-scale features at a granular level and increases the range of
receptive fields for each network layer. Res2Net block can be plugged into the state-of-the-art backbone CNN models,
e.g., ResNet, ResNeXt, and DLA. Ablation studies and experimental results on representative computer vision tasks, i.e.,
object detection, class activation mapping, and salient object detection, verify the superiority of the Res2Net over the
state-of-the-art baseline methods such as ResNet-50, DLA-60 and etc.

<p align="center">
  <img src="https://user-images.githubusercontent.com/121591093/210049799-ee3971d5-fad9-41d2-a8cd-ef64aa9d4724.png" width=800 />
</p>
<p align="center">
  <em>Figure 1. Architecture of Res2Net [<a href="#references">1</a>] </em>
</p>

## Requirements
| mindspore | ascend driver |  firmware   | cann toolkit/kernel |
| :-------: | :-----------: | :---------: | :-----------------: |
|   2.5.0   |   24.1.0      | 7.5.0.3.220 |     8.0.0.beta1     |



## Quick Start

### Preparation

#### Installation

Please refer to the [installation instruction](https://mindspore-lab.github.io/mindcv/installation/) in MindCV.

#### Dataset Preparation

Please download the [ImageNet-1K](https://www.image-net.org/challenges/LSVRC/2012/index.php) dataset for model training
and validation.

### Training

* Distributed Training

It is easy to reproduce the reported results with the pre-defined training recipe. For distributed training on multiple
Ascend 910 devices, please run

```shell
# distributed training on multiple NPU devices
msrun --bind_core=True --worker_num 8 python train.py --config configs/res2net/res2net_50_ascend.yaml --data_dir /path/to/imagenet
```




For detailed illustration of all hyper-parameters, please refer
to [config.py](https://github.com/mindspore-lab/mindcv/blob/main/config.py).

**Note:**  As the global batch size  (batch_size x num_devices) is an important hyper-parameter, it is recommended to
keep the global batch size unchanged for reproduction or adjust the learning rate linearly to a new global batch size.

* Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on single NPU device
python train.py --config configs/res2net/res2net_50_ascend.yaml --data_dir /path/to/imagenet --distribute False
```

### Validation

To validate the accuracy of the trained model, you can use `validate.py` and parse the checkpoint path
with `--ckpt_path`.

```shell
python validate.py -c configs/res2net/res2net_50_ascend.yaml --data_dir /path/to/imagenet --ckpt_path /path/to/ckpt
```

## Performance

Our reproduced model performance on ImageNet-1K is reported as follows.

Experiments are tested on ascend 910* with mindspore 2.5.0 graph mode.




| model name | params(M) | cards | batch size | resolution | jit level | graph compile | ms/step | img/s   | acc@top1 | acc@top5 | recipe                                                                                           | weight                                                                                                 |
| ---------- | --------- | ----- | ---------- | ---------- | --------- | ------------- |---------|---------| -------- | -------- | ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------ |
| res2net50  | 25.76     | 8     | 32         | 224x224    | O2        | 174s          | 40.63   | 6300.76 | 79.33    | 94.64    | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/res2net/res2net_50_ascend.yaml) | [weights](https://download-mindspore.osinfra.cn/toolkits/mindcv/res2net/res2net50-aa758355-910v2.ckpt) |



### Notes

- top-1 and top-5: Accuracy reported on the validation set of ImageNet-1K.

## References

[1] Gao S H, Cheng M M, Zhao K, et al. Res2net: A new multi-scale backbone architecture[J]. IEEE transactions on pattern
analysis and machine intelligence, 2019, 43(2): 652-662.
