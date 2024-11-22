# ReXNet

> [ReXNet: Rethinking Channel Dimensions for Efficient Model Design](https://arxiv.org/abs/2007.00992)

## Requirements
| mindspore | ascend driver |  firmware   | cann toolkit/kernel |
| :-------: | :-----------: | :---------: | :-----------------: |
|   2.3.1   |   24.1.RC2    | 7.3.0.1.231 |    8.0.RC2.beta1    |

## Introduction

ReXNets is a new model achieved based on parameterization. It utilizes a new search method for a channel configuration
via piece-wise linear functions of block index. The search space contains the conventions, and an effective channel
configuration that can be parameterized by a linear function of the block index is used. ReXNets outperforms the recent
lightweight models including NAS-based models and further showed remarkable fine-tuning performances on COCO object
detection, instance segmentation, and fine-grained classifications.

## Performance

Our reproduced model performance on ImageNet-1K is reported as follows.

- Experiments are tested on ascend 910* with mindspore 2.3.1 graph mode

<div align="center">


| model name | params(M) | cards | batch size | resolution | jit level | graph compile | ms/step | img/s   | acc@top1 | acc@top5 | recipe                                                                                          | weight                                                                                                |
| ---------- | --------- | ----- | ---------- | ---------- | --------- | ------------- | ------- | ------- | -------- | -------- | ----------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| rexnet_09  | 4.13      | 8     | 64         | 224x224    | O2        | 515s          | 115.61  | 3290.28 | 76.14    | 92.96    | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/rexnet/rexnet_x09_ascend.yaml) | [weights](https://download-mindspore.osinfra.cn/toolkits/mindcv/rexnet/rexnet_09-00223eb4-910v2.ckpt) |

</div>

- Experiments are tested on ascend 910 with mindspore 2.3.1 graph mode

<div align="center">


| model name | params(M) | cards | batch size | resolution | jit level | graph compile | ms/step | img/s   | acc@top1 | acc@top5 | recipe                                                                                          | weight                                                                                  |
| ---------- | --------- | ----- | ---------- | ---------- | --------- | ------------- | ------- | ------- | -------- | -------- | ----------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| rexnet_09  | 4.13      | 8     | 64         | 224x224    | O2        | 462s          | 130.10  | 3935.43 | 77.06    | 93.41    | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/rexnet/rexnet_x09_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/rexnet/rexnet_09-da498331.ckpt) |

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
msrun --bind_core=True --worker_num 8 python train.py --config configs/rexnet/rexnet_x09_ascend.yaml --data_dir /path/to/imagenet
```




For detailed illustration of all hyper-parameters, please refer
to [config.py](https://github.com/mindspore-lab/mindcv/blob/main/config.py).

**Note:**  As the global batch size  (batch_size x num_devices) is an important hyper-parameter, it is recommended to
keep the global batch size unchanged for reproduction or adjust the learning rate linearly to a new global batch size.

* Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on single NPU device
python train.py --config configs/rexnet/rexnet_x09_ascend.yaml --data_dir /path/to/dataset --distribute False
```

### Validation

To validate the accuracy of the trained model, you can use `validate.py` and parse the checkpoint path
with `--ckpt_path`.

```shell
python validate.py -c configs/rexnet/rexnet_x09_ascend.yaml --data_dir /path/to/imagenet --ckpt_path /path/to/ckpt
```


## References

[1] Han D, Yun S, Heo B, et al. Rethinking channel dimensions for efficient model design[C]//Proceedings of the IEEE/CVF
conference on Computer Vision and Pattern Recognition. 2021: 732-741.
