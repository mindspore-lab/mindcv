# ConvNeXt
> [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)

## Requirements
| mindspore | ascend driver |  firmware   | cann toolkit/kernel |
| :-------: | :-----------: | :---------: | :-----------------: |
|   2.3.1   |   24.1.RC2    | 7.3.0.1.231 |    8.0.RC2.beta1    |

## Introduction

In this work, the authors reexamine the design spaces and test the limits of what a pure ConvNet can achieve.
The authors gradually "modernize" a standard ResNet toward the design of a vision Transformer, and discover several key
components that contribute to the performance difference along the way. The outcome of this exploration is a family of
pure ConvNet models dubbed ConvNeXt. Constructed entirely from standard ConvNet modules, ConvNeXts compete favorably
with Transformers in terms of accuracy and scalability, achieving 87.8% ImageNet top-1 accuracy, while maintaining the
simplicity and efficiency of standard ConvNets.[[1](#references)]

<p align="center">
  <img src="https://user-images.githubusercontent.com/53842165/223907142-3bf6acfb-080a-49f5-b021-233e003318c3.png" width=250 />
</p>
<p align="center">
  <em>Figure 1. Architecture of ConvNeXt [<a href="#references">1</a>] </em>
</p>

## Performance

Our reproduced model performance on ImageNet-1K is reported as follows.

- Experiments are tested on ascend 910* with mindspore 2.3.1 graph mode

<div align="center">


| model name    | params(M) | cards | batch size | resolution | jit level | graph compile | ms/step | img/s   | acc@top1 | acc@top5 | recipe                                                                                               | weight                                                                                                      |
| ------------- | --------- | ----- | ---------- | ---------- | --------- | ------------- | ------- | ------- | -------- | -------- | ---------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| convnext_tiny | 28.59     | 8     | 16         | 224x224    | O2        | 137s          | 48.7    | 2612.24 | 81.28    | 95.61    | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/convnext/convnext_tiny_ascend.yaml) | [weights](https://download-mindspore.osinfra.cn/toolkits/mindcv/convnext/convnext_tiny-db11dc82-910v2.ckpt) |


</div>

- Experiments are tested on ascend 910 with mindspore 2.3.1 graph mode

<div align="center">


| model name    | params(M) | cards | batch size | resolution | jit level | graph compile | ms/step | img/s   | acc@top1 | acc@top5 | recipe                                                                                               | weight                                                                                        |
| ------------- | --------- | ----- | ---------- | ---------- | --------- | ------------- | ------- | ------- | -------- | -------- | ---------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| convnext_tiny | 28.59     | 8     | 16         | 224x224    | O2        | 127s          | 66.79   | 1910.45 | 81.91    | 95.79    | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/convnext/convnext_tiny_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/convnext/convnext_tiny-ae5ff8d7.ckpt) |


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
msrun --bind_core=True --worker_num 8 python train.py --config configs/convnext/convnext_tiny_ascend.yaml --data_dir /path/to/imagenet
```



For detailed illustration of all hyper-parameters, please refer to [config.py](https://github.com/mindspore-lab/mindcv/blob/main/config.py).

**Note:**  As the global batch size  (batch_size x num_devices) is an important hyper-parameter, it is recommended to keep the global batch size unchanged for reproduction or adjust the learning rate linearly to a new global batch size.

* Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on single NPU device
python train.py --config configs/convnext/convnext_tiny_ascend.yaml --data_dir /path/to/dataset --distribute False
```

### Validation

To validate the accuracy of the trained model, you can use `validate.py` and parse the checkpoint path with `--ckpt_path`.

```shell
python validate.py -c configs/convnext/convnext_tiny_ascend.yaml --data_dir /path/to/imagenet --ckpt_path /path/to/ckpt
```

## References

[1] Liu Z, Mao H, Wu C Y, et al. A convnet for the 2020s[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022: 11976-11986.
