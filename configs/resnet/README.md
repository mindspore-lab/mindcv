# ResNet

> [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)



## Introduction

Resnet is a residual learning framework to ease the training of networks that are substantially deeper than those used
previously, which is explicitly reformulated that the layers are learning residual functions with reference to the layer
inputs, instead of learning unreferenced functions. Lots of comprehensive empirical evidence showing that these residual
networks are easier to optimize, and can gain accuracy from considerably increased depth.

<p align="center">
  <img src="https://user-images.githubusercontent.com/121591093/210049796-79b70294-4250-4d0f-b078-2471880884cf.png" width=800 />
</p>
<p align="center">
  <em>Figure 1. Architecture of ResNet [<a href="#references">1</a>] </em>
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
msrun --bind_core=True --worker_num 8 python train.py --config configs/resnet/resnet_18_ascend.yaml --data_dir /path/to/imagenet
```




For detailed illustration of all hyper-parameters, please refer
to [config.py](https://github.com/mindspore-lab/mindcv/blob/main/config.py).

**Note:**  As the global batch size  (batch_size x num_devices) is an important hyper-parameter, it is recommended to
keep the global batch size unchanged for reproduction or adjust the learning rate linearly to a new global batch size.

* Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on single NPU device
python train.py --config configs/resnet/resnet_18_ascend.yaml --data_dir /path/to/dataset --distribute False
```

### Validation

To validate the accuracy of the trained model, you can use `validate.py` and parse the checkpoint path
with `--ckpt_path`.

```shell
python validate.py -c configs/resnet/resnet_18_ascend.yaml --data_dir /path/to/imagenet --ckpt_path /path/to/ckpt
```

## Performance

Our reproduced model performance on ImageNet-1K is reported as follows.

Experiments are tested on ascend 910* with mindspore 2.5.0 graph mode.




| model name | params(M) | cards | batch size | resolution | jit level | graph compile | ms/step | img/s   | acc@top1 | acc@top5 | recipe                                                                                         | weight                                                                                               |
| ---------- | --------- | ----- | ---------- | ---------- | --------- | ------------- |---------| ------- | -------- | -------- | ---------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| resnet50   | 25.61     | 8     | 32         | 224x224    | O2        | 77s           | 33.09   | 7736.47 | 76.76    | 93.31    | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/resnet/resnet_50_ascend.yaml) | [weights](https://download-mindspore.osinfra.cn/toolkits/mindcv/resnet/resnet50-f369a08d-910v2.ckpt) |



### Notes

- top-1 and top-5: Accuracy reported on the validation set of ImageNet-1K.

## References

[1] He K, Zhang X, Ren S, et al. Deep residual learning for image recognition[C]//Proceedings of the IEEE conference on
computer vision and pattern recognition. 2016: 770-778.
