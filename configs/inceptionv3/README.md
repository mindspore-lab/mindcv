# InceptionV3
> [InceptionV3: Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/pdf/1512.00567.pdf)

## Introduction

InceptionV3 is an upgraded version of GoogLeNet. One of the most important improvements of V3 is Factorization, which
decomposes 7x7 into two one-dimensional convolutions (1x7, 7x1), and 3x3 is the same (1x3, 3x1), such benefits, both It
can accelerate the calculation (excess computing power can be used to deepen the network), and can split 1 conv into 2
convs, which further increases the network depth and increases the nonlinearity of the network. It is also worth noting
that the network input from 224x224 has become 299x299, and 35x35/17x17/8x8 modules are designed more precisely. In
addition, V3 also adds batch normalization, which makes the model converge more quickly, which plays a role in partial
regularization and effectively reduces overfitting.[[1](#references)]

<p align="center">
  <img src="https://user-images.githubusercontent.com/53842165/210745725-736bd456-4d31-4f48-b958-75a53cf30e99.jpg" width=800 />
</p>
<p align="center">
  <em>Figure 1. Architecture of InceptionV3 [<a href="#references">1</a>] </em>
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
msrun --bind_core=True --worker_num 8 python train.py --config configs/inceptionv3/inception_v3_ascend.yaml --data_dir /path/to/imagenet
```




For detailed illustration of all hyper-parameters, please refer to [config.py](https://github.com/mindspore-lab/mindcv/blob/main/config.py).

**Note:**  As the global batch size  (batch_size x num_devices) is an important hyper-parameter, it is recommended to keep the global batch size unchanged for reproduction or adjust the learning rate linearly to a new global batch size.

* Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on single NPU device
python train.py --config configs/inceptionv3/inception_v3_ascend.yaml --data_dir /path/to/dataset --distribute False
```

### Validation

To validate the accuracy of the trained model, you can use `validate.py` and parse the checkpoint path with `--ckpt_path`.

```shell
python validate.py -c configs/inceptionv3/inception_v3_ascend.yaml --data_dir /path/to/imagenet --ckpt_path /path/to/ckpt
```

## Performance

Our reproduced model performance on ImageNet-1K is reported as follows.

Experiments are tested on ascend 910* with mindspore 2.5.0 graph mode.

| model name   | params(M) | cards | batch size | resolution | jit level | graph compile | ms/step | img/s   | acc@top1 | acc@top5 | recipe                                                                                                 | weight                                                                                                         |
| ------------ | --------- | ----- | ---------- | ---------- | --------- | ------------- | ------- | ------- | -------- | -------- | ------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------- |
| inception_v3 | 27.20     | 8     | 32         | 299x299    | O2        | 172s          | 70.83   | 3614.29 | 79.25    | 94.47    | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/inceptionv3/inception_v3_ascend.yaml) | [weights](https://download-mindspore.osinfra.cn/toolkits/mindcv/inception_v3/inception_v3-61a8e9ed-910v2.ckpt) |

### Notes
- top-1 and top-5: Accuracy reported on the validation set of ImageNet-1K.

## References

[1] Szegedy C, Vanhoucke V, Ioffe S, et al. Rethinking the inception architecture for computer vision[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2016: 2818-2826.
