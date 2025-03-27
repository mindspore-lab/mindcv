# Pyramid Vision Transformer

> [Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions](https://arxiv.org/abs/2102.12122)



## Introduction

PVT is a general backbone network for dense prediction without convolution operation. PVT introduces a pyramid structure
in Transformer to generate multi-scale feature maps for dense prediction tasks. PVT uses a gradual reduction strategy to
control the size of the feature maps through the patch embedding layer, and proposes a spatial reduction attention (SRA)
layer to replace the traditional multi head attention layer in the encoder, which greatly reduces the computing/memory
overhead.[[1](#References)]

![PVT](https://user-images.githubusercontent.com/74176172/210046926-2322161b-a963-4603-b3cb-86ecdca41262.png)

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

- Distributed Training

It is easy to reproduce the reported results with the pre-defined training recipe. For distributed training on multiple
Ascend 910 devices, please run

```shell
# distributed training on multiple NPU devices
msrun --bind_core=True --worker_num 8 python train.py --config configs/pvt/pvt_tiny_ascend.yaml --data_dir /path/to/imagenet
```

> If use Ascend 910 devices, need to open SATURATION_MODE via `export MS_ASCEND_CHECK_OVERFLOW_MODE="SATURATION_MODE"`


For detailed illustration of all hyper-parameters, please refer
to [config.py](https://github.com/mindspore-lab/mindcv/blob/main/config.py).

**Note:** As the global batch size (batch_size x num_devices) is an important hyper-parameter, it is recommended to keep
the global batch size unchanged for reproduction or adjust the learning rate linearly to a new global batch size.

* Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on single NPU device
python train.py --config configs/pvt/pvt_tiny_ascend.yaml --data_dir /path/to/imagenet --distribute False
```

> If use Ascend 910 devices, need to open SATURATION_MODE via `export MS_ASCEND_CHECK_OVERFLOW_MODE="SATURATION_MODE"`

### Validation

To validate the accuracy of the trained model, you can use `validate.py` and parse the checkpoint path
with `--ckpt_path`.

```shell
python validate.py --model=pvt_tiny --data_dir /path/to/imagenet --ckpt_path /path/to/ckpt
```

## Performance

Our reproduced model performance on ImageNet-1K is reported as follows.

Experiments are tested on ascend 910* with mindspore 2.5.0 graph mode.

| model name | params(M) | cards | batch size | resolution | jit level | graph compile | ms/step | img/s   | acc@top1 | acc@top5 | recipe                                                                                     | weight                                                                                            |
| ---------- | --------- | ----- | ---------- | ---------- | --------- | ------------- | ------- | ------- | -------- | -------- | ------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------- |
| pvt_tiny   | 13.23     | 8     | 128        | 224x224    | O2        | 212s          | 237.5   | 4311.58 | 74.88    | 92.12    | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/pvt/pvt_tiny_ascend.yaml) | [weights](https://download-mindspore.osinfra.cn/toolkits/mindcv/pvt/pvt_tiny-6676051f-910v2.ckpt) |


### Notes

- top-1 and top-5: Accuracy reported on the validation set of ImageNet-1K.

## References

[1]. Wang W, Xie E, Li X, et al. Pyramid vision transformer: A versatile backbone for dense prediction without
convolutions[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021: 568-578.
