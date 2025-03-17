# ShuffleNetV1

> [ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](https://arxiv.org/abs/1707.01083)



## Introduction

ShuffleNet is a computationally efficient CNN model proposed by KuangShi Technology in 2017, which, like MobileNet and
SqueezeNet, etc., is mainly intended to be applied to mobile. ShuffleNet uses two operations at its core: pointwise
group convolution and channel shuffle, which greatly reduces the model computation while maintaining accuracy.
ShuffleNet designs more efficient network structures to achieve smaller and faster models, instead of compressing or
migrating a large trained model.

<p align="center">
  <img src="https://user-images.githubusercontent.com/121591093/210049793-562b41bf-fc38-4c33-8144-5bfe75a88375.png" width=800 />
</p>
<p align="center">
  <em>Figure 1. Architecture of ShuffleNetV1 [<a href="#references">1</a>] </em>
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
msrun --bind_core=True --worker_num 8 python train.py --config configs/shufflenetv1/shufflenet_v1_0.5_ascend.yaml --data_dir /path/to/imagenet
```




For detailed illustration of all hyper-parameters, please refer
to [config.py](https://github.com/mindspore-lab/mindcv/blob/main/config.py).

**Note:**  As the global batch size  (batch_size x num_devices) is an important hyper-parameter, it is recommended to
keep the global batch size unchanged for reproduction or adjust the learning rate linearly to a new global batch size.

* Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on single NPU device
python train.py --config configs/shufflenetv1/shufflenet_v1_0.5_ascend.yaml --data_dir /path/to/dataset --distribute False
```

### Validation

To validate the accuracy of the trained model, you can use `validate.py` and parse the checkpoint path
with `--ckpt_path`.

```shell
python validate.py -c configs/shufflenetv1/shufflenet_v1_0.5_ascend.yaml --data_dir /path/to/imagenet --ckpt_path /path/to/ckpt
```

## Performance

Our reproduced model performance on ImageNet-1K is reported as follows.

Experiments are tested on ascend 910* with mindspore 2.5.0 graph mode.




| model name          | params(M) | cards | batch size | resolution | jit level | graph compile | ms/step | img/s    | acc@top1 | acc@top5 | recipe                                                                                                       | weight                                                                                                                           |
| ------------------- | --------- | ----- | ---------- | ---------- | --------- |---------------| ------- | -------- | -------- | -------- | ------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------- |
| shufflenet_v1_g3_05 | 0.73      | 8     | 64         | 224x224    | O2        | 191s          | 47.77   | 10718.02 | 57.08    | 79.89    | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/shufflenetv1/shufflenet_v1_0.5_ascend.yaml) | [weights](https://download-mindspore.osinfra.cn/toolkits/mindcv/shufflenet/shufflenetv1/shufflenet_v1_g3_05-56209ef3-910v2.ckpt) |



Experiments are tested on ascend 910 with mindspore 2.5.0 graph mode.




| model name          | params(M) | cards | batch size | resolution | jit level | graph compile | ms/step | img/s    | acc@top1 | acc@top5 | recipe                                                                                                       | weight                                                                                                             |
| ------------------- | --------- | ----- | ---------- | ---------- | --------- | ------------- | ------- | -------- | -------- | -------- | ------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------ |
| shufflenet_v1_g3_05 | 0.73      | 8     | 64         | 224x224    | O2        | 169s          | 40.62   | 12604.63 | 57.05    | 79.73    | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/shufflenetv1/shufflenet_v1_0.5_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/shufflenet/shufflenetv1/shufflenet_v1_g3_05-42cfe109.ckpt) |



### Notes

- top-1 and top-5: Accuracy reported on the validation set of ImageNet-1K.

## References

[1] Zhang X, Zhou X, Lin M, et al. Shufflenet: An extremely efficient convolutional neural network for mobile devices[C]
//Proceedings of the IEEE conference on computer vision and pattern recognition. 2018: 6848-6856.
