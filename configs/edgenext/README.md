# EdgeNeXt

> [EdgeNeXt: Efficiently Amalgamated CNN-Transformer Architecture for Mobile Vision Applications](https://arxiv.org/abs/2206.10589)

## Introduction

EdgeNeXt effectively combines the strengths of both CNN and Transformer models and is a
new efficient hybrid architecture. EdgeNeXt introduces a split depth-wise transpose
attention (SDTA) encoder that splits input tensors into multiple channel groups and
utilizes depth-wise convolution along with self-attention across channel dimensions
to implicitly increase the receptive field and encode multi-scale features.[[1](#references)]

<p align="center">
  <img src="https://user-images.githubusercontent.com/52945530/210045582-d31f832d-22e0-47bd-927f-74cf2daed91a.png" width=800 />
</p>
<p align="center">
  <em>Figure 1. Architecture of EdgeNeXt [<a href="#references">1</a>] </em>
</p>

## Results

Our reproduced model performance on ImageNet-1K is reported as follows.

- ascend 910* with graph mode

<div align="center">


| model             | top-1 (%) | top-5 (%) | params (M) | batch size | cards | ms/step | jit_level | recipe                                                                                                   | download                                                                                                        |
| ----------------- | --------- | --------- | ---------- | ---------- | ----- | ------- | --------- | -------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| edgenext_xx_small | 70.64     | 89.75     | 1.33       | 256        | 8     | 239.38  | O2        | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/edgenext/edgenext_xx_small_ascend.yaml) | [weights](https://download-mindspore.osinfra.cn/toolkits/mindcv/edgenext/edgenext_xx_small-cad13d2c-910v2.ckpt) |


</div>

- ascend 910 with graph mode

<div align="center">


| model             | top-1 (%) | top-5 (%) | params (M) | batch size | cards | ms/step | jit_level | recipe                                                                                                   | download                                                                                          |
| ----------------- | --------- | --------- | ---------- | ---------- | ----- | ------- | --------- | -------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| edgenext_xx_small | 71.02     | 89.99     | 1.33       | 256        | 8     | 191.24  | O2        | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/edgenext/edgenext_xx_small_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/edgenext/edgenext_xx_small-afc971fb.ckpt) |


</div>

#### Notes
- Top-1 and Top-5: Accuracy reported on the validation set of ImageNet-1K.

## Quick Start

### Preparation

#### Installation

Please refer to the [installation instruction](https://mindspore-lab.github.io/mindcv/installation/) in MindCV.

#### Dataset Preparation

Please download the [ImageNet-1K](https://www.image-net.org/challenges/LSVRC/2012/index.php) dataset for model training and validation.

### Training

* Distributed Training

It is easy to reproduce the reported results with the pre-defined training recipe. For distributed training on multiple Ascend 910 devices, please run

```shell
# distributed training on multiple NPU devices
msrun --bind_core=True --worker_num 8 python train.py --config configs/edgenext/edgenext_small_ascend.yaml --data_dir /path/to/imagenet
```



For detailed illustration of all hyper-parameters, please refer to [config.py](https://github.com/mindspore-lab/mindcv/blob/main/config.py).

**Note:**  As the global batch size  (batch_size x num_devices) is an important hyper-parameter, it is recommended to keep the global batch size unchanged for reproduction or adjust the learning rate linearly to a new global batch size.

* Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on single NPU device
python train.py --config configs/edgenext/edgenext_small_ascend.yaml --data_dir /path/to/dataset --distribute False
```

### Validation

To validate the accuracy of the trained model, you can use `validate.py` and parse the checkpoint path with `--ckpt_path`.

```
python validate.py -c configs/edgenext/edgenext_small_ascend.yaml --data_dir /path/to/imagenet --ckpt_path /path/to/ckpt
```

## References

<!--- Guideline: Citation format should follow GB/T 7714. -->
[1] Maaz M, Shaker A, Cholakkal H, et al. EdgeNeXt: efficiently amalgamated CNN-transformer architecture for Mobile vision applications[J]. arXiv preprint arXiv:2206.10589, 2022.
