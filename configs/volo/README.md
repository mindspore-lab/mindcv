# VOLO

> [VOLO: Vision Outlooker for Visual Recognition ](https://arxiv.org/abs/2106.13112)



## Introduction

Vision Outlooker (VOLO), a novel outlook attention, presents a simple and general architecture. Unlike self-attention
that focuses on global dependency modeling at a coarse level, the outlook attention efficiently encodes finer-level
features and contexts into tokens, which is shown to be critically beneficial to recognition performance but largely
ignored by the self-attention. Five versions different from model scaling are introduced based on the proposed VOLO:
VOLO-D1 with 27M parameters to VOLO-D5 with 296M. Experiments show that the best one, VOLO-D5, achieves 87.1% top-1
accuracy on ImageNet-1K classification, which is the first model exceeding 87% accuracy on this competitive benchmark,
without using any extra training data.

<p align="center">
  <img src="https://user-images.githubusercontent.com/61639773/249760556-b7aa4b23-a204-4061-8bed-170b02c52419.png" width=800 />
</p>
<p align="center">
  <em>Figure 1. Illustration of outlook attention. [<a href="#references">1</a>] </em>
</p>

## Requirements
| mindspore | ascend driver |  firmware   | cann toolkit/kernel |
| :-------: | :-----------: | :---------: | :-----------------: |
|   2.5.0   |   24.1.0      | 7.5.0.3.220 |     8.0.0.beta1     |



## Quick Start

### Preparation

#### Installation

Please refer to the [installation instruction](https://github.com/mindspore-lab/mindcv#installation) in MindCV.

#### Dataset Preparation

Please download the [ImageNet-1K](https://www.image-net.org/challenges/LSVRC/2012/index.php) dataset for model training
and validation.

### Training

* Distributed Training

It is easy to reproduce the reported results with the pre-defined training recipe. For distributed training on multiple
Ascend 910 devices, please run

```shell
# distributed training on multiple Ascend devices
msrun --bind_core=True --worker_num 8 python train.py --config configs/volo/volo_d1_ascend.yaml --data_dir /path/to/imagenet
```


For detailed illustration of all hyper-parameters, please refer
to [config.py](https://github.com/mindspore-lab/mindcv/blob/main/config.py).

**Note:** As the global batch size (batch_size x num_devices) is an important hyper-parameter, it is recommended to keep
the global batch size unchanged for reproduction or adjust the learning rate linearly to a new global batch size.

- Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on a CPU/GPU/Ascend device
python train.py --config configs/volo/volo_d1_ascend.yaml --data_dir /path/to/dataset --distribute False
```

### Validation

To validate the accuracy of the trained model, you can use `validate.py` and parse the checkpoint path
with `--ckpt_path`.

```shell
python validate.py -c configs/volo/volo_d1_ascend.yaml --data_dir /path/to/imagenet --ckpt_path /path/to/ckpt
```

## Performance

Our reproduced model performance on ImageNet-1K is reported as follows.

Experiments are tested on Ascend Atlas 800T A2 machines with mindspore 2.5.0 graph mode.

| model name | params(M) | cards |  batch size  |  resolution  |  jit level  |  graph compile  |  ms/step  |   img/s   |  acc@top1  |  acc@top5  |                                               recipe                                               |                                                  weight                                                   |
|:----------:|:---------:|:-----:|:------------:|:------------:|:-----------:|:---------------:|:---------:|:---------:|:----------:|:----------:|:--------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------:|
|  volo_d1   |   26.63   |   8   | 128        | 224x224    | O2        | 368s          | 230.05  | 4451.21 | 82.97    | 96.21    | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/volo/volo_d1_ascend.yaml) | [weights](https://download-mindspore.osinfra.cn/toolkits/mindcv/volo/volo_d1-177_1251_v2.ckpt)           |

### Notes

- top-1 and top-5: Accuracy reported on the validation set of ImageNet-1K.


## References

[1] Yuan L , Hou Q , Jiang Z , et al. VOLO: Vision Outlooker for Visual Recognition[J]. . arXiv preprint arXiv:
2106.13112, 2021.
