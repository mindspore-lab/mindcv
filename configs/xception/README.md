# Xception

> [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/pdf/1610.02357.pdf)



## Introduction

Xception is another improved network of InceptionV3 in addition to inceptionV4, using a deep convolutional neural
network architecture involving depthwise separable convolution, which was developed by Google researchers. Google
interprets the Inception module in convolutional neural networks as an intermediate step between regular convolution and
depthwise separable convolution operations. From this point of view, the depthwise separable convolution can be
understood as having the largest number of Inception modules, that is, the extreme idea proposed in the paper, combined
with the idea of residual network, Google proposed a new type of deep convolutional neural network inspired by Inception
Network architecture where the Inception module has been replaced by a depthwise separable convolution
module.[[1](#references)]

<p align="center">
  <img src="https://user-images.githubusercontent.com/53842165/210751172-90b49732-33d1-4e68-adf7-6881b07a3c54.jpg" width=800 />
</p>
<p align="center">
  <em>Figure 1. Architecture of Xception [<a href="#references">1</a>] </em>
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
msrun --bind_core=True --worker_num 8 python train.py --config configs/xception/xception_ascend.yaml --data_dir /path/to/imagenet
```




For detailed illustration of all hyper-parameters, please refer
to [config.py](https://github.com/mindspore-lab/mindcv/blob/main/config.py).

**Note:**  As the global batch size  (batch_size x num_devices) is an important hyper-parameter, it is recommended to
keep the global batch size unchanged for reproduction or adjust the learning rate linearly to a new global batch size.

* Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on single NPU device
python train.py --config configs/xception/xception_ascend.yaml --data_dir /path/to/dataset --distribute False
```

### Validation

To validate the accuracy of the trained model, you can use `validate.py` and parse the checkpoint path
with `--ckpt_path`.

```shell
python validate.py -c configs/xception/xception_ascend.yaml --data_dir /path/to/imagenet --ckpt_path /path/to/ckpt
```

## Performance

Our reproduced model performance on ImageNet-1K is reported as follows.

Experiments are tested on ascend 910* with mindspore 2.5.0 graph mode.

| model name  | cards | batch size | resolution | jit level | graph compile | ms/step | img/s   | recipe                                                                                           | weight                                                                                                  | acc@top1 | acc@top5 |
| ----------- | ----- | ---------- | ---------- | --------- | ------------- | ------- | ------- | ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------- | -------- | -------- |
| xception   | 8     | 32         | 224x224    | O2        | 186s          | 83.40   | 3069.54 | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/xception/xception_ascend.yaml)   | [weights](https://download-mindspore.osinfra.cn/toolkits/mindcv/xception/xception-174_5004_v2.ckpt)   | 76.31    | 92.80    |


### Notes

- top-1 and top-5: Accuracy reported on the validation set of ImageNet-1K.


## References

[1] Chollet F. Xception: Deep learning with depthwise separable convolutions[C]//Proceedings of the IEEE conference on
computer vision and pattern recognition. 2017: 1251-1258.
