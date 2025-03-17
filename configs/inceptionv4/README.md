# InceptionV4
> [InceptionV4: Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/pdf/1602.07261.pdf)

## Introduction

InceptionV4 studies whether the Inception module combined with Residual Connection can be improved. It is found that the
structure of ResNet can greatly accelerate the training, and the performance is also improved. An Inception-ResNet v2
network is obtained, and a deeper and more optimized Inception v4 model is also designed, which can achieve comparable
performance with Inception-ResNet v2.[[1](#references)]

<p align="center">
  <img src="https://user-images.githubusercontent.com/53842165/210749903-5ff23c0e-547f-487d-bb64-70b6e99031ea.jpg" width=500 />
</p>
<p align="center">
  <em>Figure 1. Architecture of InceptionV4 [<a href="#references">1</a>] </em>
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
Please download the [ImageNet-1K](https://www.image-net.org/challenges/LSVRC/2012/index.php) dataset for model training and validation.

### Training

* Distributed Training

It is easy to reproduce the reported results with the pre-defined training recipe. For distributed training on multiple Ascend 910 devices, please run

```shell
# distributed training on multiple NPU devices
msrun --bind_core=True --worker_num 8 python train.py --config configs/inceptionv4/inception_v4_ascend.yaml --data_dir /path/to/imagenet
```




For detailed illustration of all hyper-parameters, please refer to [config.py](https://github.com/mindspore-lab/mindcv/blob/main/config.py).

**Note:**  As the global batch size  (batch_size x num_devices) is an important hyper-parameter, it is recommended to keep the global batch size unchanged for reproduction or adjust the learning rate linearly to a new global batch size.

* Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on single NPU device
python train.py --config configs/inceptionv4/inception_v4_ascend.yaml --data_dir /path/to/dataset --distribute False
```

### Validation

To validate the accuracy of the trained model, you can use `validate.py` and parse the checkpoint path with `--ckpt_path`.

```shell
python validate.py -c configs/inceptionv4/inception_v4_ascend.yaml --data_dir /path/to/imagenet --ckpt_path /path/to/ckpt
```

## Performance

Our reproduced model performance on ImageNet-1K is reported as follows.

Experiments are tested on ascend 910* with mindspore 2.5.0 graph mode.

| model name   | params(M) | cards | batch size | resolution | jit level | graph compile | ms/step | img/s   | acc@top1 | acc@top5 | recipe                                                                                                 | weight                                                                                                         |
| ------------ | --------- | ----- | ---------- | ---------- | --------- |---------------| ------- | ------- | -------- | -------- | ------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------- |
| inception_v4 | 42.74     | 8     | 32         | 299x299    | O2        | 263s          | 80.97   | 3161.66 | 80.98    | 95.25    | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/inceptionv4/inception_v4_ascend.yaml) | [weights](https://download-mindspore.osinfra.cn/toolkits/mindcv/inception_v4/inception_v4-56e798fc-910v2.ckpt) |


Experiments are tested on ascend 910 with mindspore 2.5.0 graph mode.

| model name   | params(M) | cards | batch size | resolution | jit level | graph compile | ms/step | img/s   | acc@top1 | acc@top5 | recipe                                                                                                 | weight                                                                                           |
| ------------ | --------- | ----- | ---------- | ---------- | --------- | ------------- | ------- | ------- | -------- | -------- | ------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------ |
| inception_v4 | 42.74     | 8     | 32         | 299x299    | O2        | 177s          | 76.19   | 3360.02 | 80.88    | 95.34    | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/inceptionv4/inception_v4_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/inception_v4/inception_v4-db9c45b3.ckpt) |

### Notes
- top-1 and top-5: Accuracy reported on the validation set of ImageNet-1K.

## References

[1] Szegedy C, Ioffe S, Vanhoucke V, et al. Inception-v4, inception-resnet and the impact of residual connections on learning[C]//Thirty-first AAAI conference on artificial intelligence. 2017.
