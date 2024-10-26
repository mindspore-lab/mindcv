# PVTV2

> [PVT v2: Improved Baselines with Pyramid Vision Transformer](https://arxiv.org/abs/2106.13797)

## Introduction

In this work, the authors present new baselines by improving the original Pyramid Vision Transformer (PVT v1) by adding
three designs, including (1) linear complexity attention layer, (2) overlapping patch embedding, and (3) convolutional
feed-forward network. With these modifications, PVT v2 reduces the computational complexity of PVT v1 to linear and
achieves significant improvements on fundamental vision tasks such as classification, detection, and
segmentation.[[1](#references)]

<p align="center">
  <img src="https://user-images.githubusercontent.com/53842165/219326579-de903edb-131f-4905-a3fe-7be2cb8cc8b7.png" width=500 />
</p>
<p align="center">
  <em>Figure 1. Architecture of PVTV2 [<a href="#references">1</a>] </em>
</p>

## Results

Our reproduced model performance on ImageNet-1K is reported as follows.

- ascend 910* with graph mode

<div align="center">


|   model   | top-1 (%) | top-5 (%) | params (M) | batch size | cards | ms/step | jit_level | recipe                                                                                        | download                                                                                              |
| :-------: | :-------: | :-------: | :--------: | ---------- | ----- | ------- | --------- | --------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| pvt_v2_b0 |   71.25   |   90.50   |    3.67    | 128        | 8     | 255.76  | O2        | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/pvtv2/pvt_v2_b0_ascend.yaml) | [weights](https://download-mindspore.osinfra.cn/toolkits/mindcv/pvt_v2/pvt_v2_b0-d9cd9d6a-910v2.ckpt) |

</div>

- ascend 910 with graph mode

<div align="center">


|   model   | top-1 (%) | top-5 (%) | params (M) | batch size | cards | ms/step | jit_level | recipe                                                                                        | download                                                                                |
| :-------: | :-------: | :-------: | :--------: | ---------- | ----- | ------- | --------- | --------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| pvt_v2_b0 |   71.50   |   90.60   |    3.67    | 128        | 8     | 269.38  | O2        | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/pvtv2/pvt_v2_b0_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/pvt_v2/pvt_v2_b0-1c4f6683.ckpt) |

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
# distrubted training on multiple GPU/Ascend devices
msrun --bind_core=True --worker_num 8 python train.py --config configs/pvtv2/pvt_v2_b0_ascend.yaml --data_dir /path/to/imagenet
```




For detailed illustration of all hyper-parameters, please refer
to [config.py](https://github.com/mindspore-lab/mindcv/blob/main/config.py).

**Note:**  As the global batch size  (batch_size x num_devices) is an important hyper-parameter, it is recommended to
keep the global batch size unchanged for reproduction or adjust the learning rate linearly to a new global batch size.

* Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on single NPU device
python train.py --config configs/pvtv2/pvt_v2_b0_ascend.yaml --data_dir /path/to/dataset --distribute False
```

### Validation

To validate the accuracy of the trained model, you can use `validate.py` and parse the checkpoint path
with `--ckpt_path`.

```shell
python validate.py -c configs/pvtv2/pvt_v2_b0_ascend.yaml --data_dir /path/to/imagenet --ckpt_path /path/to/ckpt
```


## References

[1] Wang W, Xie E, Li X, et al. Pvt v2: Improved baselines with pyramid vision transformer[J]. Computational Visual
Media, 2022, 8(3): 415-424.
