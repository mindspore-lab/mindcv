# MobileNetV3
> [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)



## Introduction

MobileNet v3 was published in 2019, and this v3 version combines the deep separable convolution of v1, the Inverted Residuals and Linear Bottleneck of v2, and the SE module to search the configuration and parameters of the network using NAS (Neural Architecture Search).MobileNetV3 first uses MnasNet to perform a coarse structure search, and then uses reinforcement learning to select the optimal configuration from a set of discrete choices. Afterwards, MobileNetV3 then fine-tunes the architecture using NetAdapt, which exemplifies NetAdapt's complementary capability to tune underutilized activation channels with a small drop.

mobilenet-v3 offers two versions, mobilenet-v3 large and mobilenet-v3 small, for situations with different resource requirements. The paper mentions that mobilenet-v3 small, for the imagenet classification task, has an accuracy The paper mentions that mobilenet-v3 small achieves about 3.2% better accuracy and 15% less time than mobilenet-v2 for the imagenet classification task, mobilenet-v3 large achieves about 4.6% better accuracy and 5% less time than mobilenet-v2 for the imagenet classification task, mobilenet-v3 large achieves the same accuracy and 25% faster speedup in COCO compared to v2 The improvement in the segmentation algorithm is also observed.[[1](#references)]

<p align="center">
  <img src="https://user-images.githubusercontent.com/53842165/210044297-d658ca54-e6ff-4c0f-8080-88072814d8e6.png" width=800 />
</p>
<p align="center">
  <em>Figure 1. Architecture of MobileNetV3 [<a href="#references">1</a>] </em>
</p>

## Requirements
| mindspore | ascend driver |  firmware   | cann toolkit/kernel |
| :-------: | :-----------: | :---------: |:-------------------:|
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
msrun --bind_core=True --worker_num 8 python train.py --config configs/mobilenetv3/mobilenet_v3_small_ascend.yaml --data_dir /path/to/imagenet
```




For detailed illustration of all hyper-parameters, please refer to [config.py](https://github.com/mindspore-lab/mindcv/blob/main/config.py).

**Note:**  As the global batch size  (batch_size x num_devices) is an important hyper-parameter, it is recommended to keep the global batch size unchanged for reproduction or adjust the learning rate linearly to a new global batch size.

* Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on single NPU device
python train.py --config configs/mobilenetv3/mobilenet_v3_small_ascend.yaml --data_dir /path/to/dataset --distribute False
```

### Validation

To validate the accuracy of the trained model, you can use `validate.py` and parse the checkpoint path with `--ckpt_path`.

```shell
python validate.py -c configs/mobilenetv3/mobilenet_v3_small_ascend.yaml --data_dir /path/to/imagenet --ckpt_path /path/to/ckpt
```

## Performance

Our reproduced model performance on ImageNet-1K is reported as follows.

Experiments are tested on ascend 910* with mindspore 2.5.0 graph mode.


| model name             | params(M) | cards | batch size | resolution | jit level | graph compile | ms/step | img/s    | acc@top1 | acc@top5 | recipe                                                                                                       | weight                                                                                                                            |
| ---------------------- | --------- | ----- | ---------- | ---------- | --------- | ------------- |---------| -------- | -------- | -------- | ------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------- |
| mobilenet_v3_small_100 | 2.55      | 8     | 75         | 224x224    | O2        | 184s          | 54.21   | 11068.06 | 68.07    | 87.77    | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/mobilenetv3/mobilenet_v3_small_ascend.yaml) | [weights](https://download-mindspore.osinfra.cn/toolkits/mindcv/mobilenet/mobilenetv3/mobilenet_v3_small_100-6fa3c17d-910v2.ckpt) |
| mobilenet_v3_large_100 | 5.51      | 8     | 75         | 224x224    | O2        | 354s          | 56.87   | 10550.37 | 75.59    | 92.57    | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/mobilenetv3/mobilenet_v3_large_ascend.yaml) | [weights](https://download-mindspore.osinfra.cn/toolkits/mindcv/mobilenet/mobilenetv3/mobilenet_v3_large_100-bd4e7bdc-910v2.ckpt) |

### Notes
- top-1 and top-5: Accuracy reported on the validation set of ImageNet-1K.

## References

[1] Howard A, Sandler M, Chu G, et al. Searching for mobilenetv3[C]//Proceedings of the IEEE/CVF international conference on computer vision. 2019: 1314-1324.
