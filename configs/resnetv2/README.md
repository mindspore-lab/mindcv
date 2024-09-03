# ResNetV2

> [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027)

## Requirements
| mindspore | ascend driver |  firmware   | cann toolkit/kernel |
| :-------: | :-----------: | :---------: | :-----------------: |
|   2.3.1   |   24.1.RC2    | 7.3.0.1.231 |    8.0.RC2.beta1    |

## Introduction

Author analyzes the propagation formulations behind the residual building blocks, which suggest that the forward and
backward signals can be directly propagated from one block
to any other block, when using identity mappings as the skip connections and after-addition activation.

<p align="center">
  <img src="https://user-images.githubusercontent.com/52945530/224595993-ba8617da-e55d-4d19-a487-3340026393c9.png" width=300 height=400 />
</p>
<p align="center">
  <em>Figure 1. Architecture of ResNetV2 [<a href="#references">1</a>] </em>
</p>

## Performance

Our reproduced model performance on ImageNet-1K is reported as follows.

- Experiments are tested on ascend 910* with mindspore 2.3.1 graph mode

<div align="center">


| model name  | params(M) | cards | batch size | resolution | jit level | graph compile | ms/step | img/s   | acc@top1 | acc@top5 | recipe                                                                                             | weight                                                                                                    |
| ----------- | --------- | ----- | ---------- | ---------- | --------- |---------------| ------- | ------- | -------- | -------- | -------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| resnetv2_50 | 25.60     | 8     | 32         | 224x224    | O2        | 120s          | 32.19   | 7781.16 | 77.03    | 93.29    | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/resnetv2/resnetv2_50_ascend.yaml) | [weights](https://download-mindspore.osinfra.cn/toolkits/mindcv/resnetv2/resnetv2_50-a0b9f7f8-910v2.ckpt) |

</div>

- Experiments are tested on ascend 910 with mindspore 2.3.1 graph mode

<div align="center">


| model name  | params(M) | cards | batch size | resolution | jit level | graph compile | ms/step | img/s   | acc@top1 | acc@top5 | recipe                                                                                             | weight                                                                                      |
| ----------- | --------- | ----- | ---------- | ---------- | --------- | ------------- | ------- | ------- | -------- | -------- | -------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| resnetv2_50 | 25.60     | 8     | 32         | 224x224    | O2        | 52s           | 32.66   | 7838.33 | 76.90    | 93.37    | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/resnetv2/resnetv2_50_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/resnetv2/resnetv2_50-3c2f143b.ckpt) |

</div>

#### Notes

- Top-1 and Top-5: Accuracy reported on the validation set of ImageNet-1K.

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
msrun --bind_core=True --worker_num 8 python train.py --config configs/resnetv2/resnetv2_50_ascend.yaml --data_dir /path/to/imagenet
```




For detailed illustration of all hyper-parameters, please refer
to [config.py](https://github.com/mindspore-lab/mindcv/blob/main/config.py).

**Note:**  As the global batch size  (batch_size x num_devices) is an important hyper-parameter, it is recommended to
keep the global batch size unchanged for reproduction or adjust the learning rate linearly to a new global batch size.

* Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on single NPU device
python train.py --config configs/resnetv2/resnetv2_50_ascend.yaml --data_dir /path/to/dataset --distribute False
```

### Validation

To validate the accuracy of the trained model, you can use `validate.py` and parse the checkpoint path
with `--ckpt_path`.

```shell
python validate.py -c configs/resnetv2/resnetv2_50_ascend.yaml --data_dir /path/to/imagenet --ckpt_path /path/to/ckpt
```


## References

[1] He K, Zhang X, Ren S, et al. Identity mappings in deep residual networks[C]//Computer Vision–ECCV 2016: 14th
European Conference, Amsterdam, The Netherlands, October 11–14, 2016, Proceedings, Part IV 14. Springer International
Publishing, 2016: 630-645.
