<div align="center">

# MindCV

[![CI](https://github.com/mindspore-lab/mindcv/actions/workflows/ci.yml/badge.svg)](https://github.com/mindspore-lab/mindcv/actions/workflows/ci.yml)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mindcv)](https://pypi.org/project/mindcv)
[![PyPI](https://img.shields.io/pypi/v/mindcv)](https://pypi.org/project/mindcv)
[![docs](https://img.shields.io/badge/docs-latest-blue)](https://mindcv.readthedocs.io/en/latest)
[![license](https://img.shields.io/github/license/mindspore-lab/mindcv.svg)](https://github.com/mindspore-lab/mindcv/blob/main/LICENSE.md)
[![open issues](https://img.shields.io/github/issues/mindspore-lab/mindcv)](https://github.com/mindspore-lab/mindcv/issues)
[![PRs](https://img.shields.io/badge/PRs-welcome-pink.svg)](https://github.com/mindspore-lab/mindcv/pulls)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

English | [中文](README_CN.md)

[Introduction](#introduction) |
[Installation](#installation) |
[Get Started](#get-started) |
[Tutorials](#tutorials) |
[Model List](#model-list) |
[Supported Algorithms](#supported-algorithms) |
[Notes](#notes)

</div>

## Introduction
MindCV is an open-source toolbox for computer vision research and development based on [MindSpore](https://www.mindspore.cn/en). It collects a series of classic and SoTA vision models, such as ResNet and SwinTransformer, along with their pre-trained weights and training strategies. SoTA methods such as auto augmentation are also provided for performance improvement. With the decoupled module design, it is easy to apply or adapt MindCV to your own CV tasks.

<details open>
<summary> Major Features </summary>

- **Easy-to-Use.** MindCV decomposes the vision framework into various configurable components. It is easy to customize your data pipeline, models, and learning pipeline with MindCV:

```python
>>> import mindcv
# create a dataset
>>> dataset = mindcv.create_dataset('cifar10', download=True)
# create a model
>>> network = mindcv.create_model('resnet50', pretrained=True)
```

Users can customize and launch their transfer learning or training task in one command line.

``` python
# transfer learning in one command line
>>> !python train.py --model=swin_tiny --pretrained --opt=adamw --lr=0.001 --data_dir={data_dir}
```

- **State-of-The-Art.** MindCV provides various CNN-based and Transformer-based vision models including SwinTransformer. Their pretrained weights and performance reports are provided to help users select and reuse the right model:

- **Flexibility and efficiency.** MindCV is built on MindSpore which is an efficent DL framework that can be run on different hardware platforms (GPU/CPU/Ascend). It supports both graph mode for high efficiency and pynative mode for flexibility.


</details>

### Benchmark Results

The performance of the models trained with MindCV is summarized in [benchmark_results.md](./benchmark_results.md), where the training recipes and weights are both available.

Model introduction and training details can be viewed in each subfolder under [configs](configs).

## Installation

### Dependency

- mindspore >= 1.8.1
- numpy >= 1.17.0
- pyyaml >= 5.3
- tqdm
- openmpi 4.0.3 (for distributed mode)

To install the dependency, please run
```shell
pip install -r requirements.txt
```

MindSpore can be easily installed by following the official [instructions](https://www.mindspore.cn/install) where you can select your hardware platform for the best fit. To run in distributed mode, [openmpi](https://www.open-mpi.org/software/ompi/v4.0/) is required to install.

The following instructions assume the desired dependency is fulfilled.

### Install with PyPI

The released version of MindCV can be installed via `PyPI` as follows:
```shell
pip install mindcv
```

### Install from Source

The latest version of MindCV can be installed as follows:
```shell
pip install git+https://github.com/mindspore-lab/mindcv.git
```

> Notes: MindCV can be installed on Linux and Mac but not on Windows currently.

## Get Started

### Hands-on Tutorial

To get started with MindCV, please see the [transfer learning tutorial](tutorials/finetune.md), which will give you a quick tour on each key component and the train/validate/predict pipelines.

Below are a few code snippets for your taste.

```python
>>> import mindcv
# List and find a pretrained vision model
>>> mindcv.list_models("swin*", pretrained=True)
['swin_tiny']
# Create the model object
>>> network = mindcv.create_model('swin_tiny', pretrained=True)
# Validate its accuracy
>>> !python validate.py --model=swin_tiny --pretrained --dataset=imagenet --val_split=validation
{'Top_1_Accuracy': 0.808343989769821, 'Top_5_Accuracy': 0.9527253836317136, 'loss': 0.8474242982580839}
```

**Image classification demo**

<p align="left">
  <img src="https://user-images.githubusercontent.com/8156835/210049681-89f68b9f-eb44-44e2-b689-4d30c93c6191.jpg" width=360 />
</p>

Infer the input image with a pretrained SoTA model,

```python
>>> !python infer.py --model=swin_tiny --image_path='./tutorials/data/test/dog/dog.jpg'
{'Labrador retriever': 0.5700152, 'golden retriever': 0.034551315, 'kelpie': 0.010108651, 'Chesapeake Bay retriever': 0.008229004, 'Walker hound, Walker foxhound': 0.007791956}
```
The top-1 prediction result is labrador retriever (拉布拉多犬), which is the breed of this cut dog.

### Training

It is easy to train your model on a standard or customized dataset using `train.py`, where the training strategy (e.g., augmentation, LR scheduling) can be configured with external arguments or a yaml config file.

- Standalone Training

``` shell
# standalone training
python train.py --model=resnet50 --dataset=cifar10 --dataset_download
```

Above is an example for training ResNet50 on CIFAR10 dataset on a CPU/GPU/Ascend device

- Distributed Training

For large datasets like ImageNet, it is necessary to do training in distributed mode on multiple devices. This can be achieved with `mpirun` and parallel features supported by MindSpore.

```shell
# distributed training
# assume you have 4 GPUs/NPUs
mpirun -n 4 python train.py --distribute \
	--model=densenet121 --dataset=imagenet --data_dir=/path/to/imagenet
```
> Notes: If the script is executed by the root user, the `--allow-run-as-root` parameter must be added to `mpirun`.

Detailed parameter definitions  can be seen in `config.py` and checked by running `python train.py --help'.

To resume training, please set the `--ckpt_path` and `--ckpt_save_dir` arguments. The optimizer state including the learning rate of the last stopped epoch will also be recovered.

- Config and Training Strategy

You can configure your model and other components either by specifying external parameters or by writing a yaml config file. Here is an example of training using a preset yaml file.

```shell
mpirun --allow-run-as-root -n 4 python train.py -c configs/squeezenet/squeezenet_1.0_gpu.yaml
```

**Pre-defined Training Strategies:** We provide more than 20 training recipes that achieve SoTA results on ImageNet currently. Please look into the [`configs`](configs) folder for details. Please feel free to adapt these training strategies to your own model for performance improvement， which can be easily done by modifying the yaml file.

- Train on ModelArts/OpenI Platform

To run training on the [ModelArts](https://www.huaweicloud.com/intl/en-us/product/modelarts.html) or [OpenI](https://openi.pcl.ac.cn/) cloud platform:

```text
1. Create a new training task on the cloud platform.
2. Add run parameter `config` and specify the path to the yaml config file on the website UI interface.
3. Add run parameter `enable_modelarts` and set True on the website UI interface.
4. Fill in other blanks on the website and launch the training task.
```

### Validation

To evalute the model performance, please run `validate.py`

```shell
# validate a trained checkpoint
python validate.py --model=resnet50 --dataset=imagenet --data_dir=/path/to/data --ckpt_path=/path/to/model.ckpt
```

- Validation while Training

You can also track the validation accuracy during training by enabling the `--val_while_train` option.

```shell
python train.py --model=resnet50 --dataset=cifar10 \
		--val_while_train --val_split=test --val_interval=1
```

The training loss and validation accuracy for each epoch  will be saved in `{ckpt_save_dir}/results.log`.

More examples about training and validation can be seen in [examples/scripts](examples/scripts).

- Graph Mode and Pynative Mode

By default, the training pipeline `train.py` is run in [graph mode](https://www.mindspore.cn/tutorials/zh-CN/r1.8/advanced/pynative_graph/mode.html#%E9%9D%99%E6%80%81%E5%9B%BE) on MindSpore, which is optimized for efficiency and parallel computing with a compiled static graph. In contrast, [pynative mode](https://www.mindspore.cn/tutorials/zh-CN/r1.8/advanced/pynative_graph/mode.html#%E5%8A%A8%E6%80%81%E5%9B%BE) is optimized for flexibility and easy debugging. You may alter the parameter `--mode` to switch to pure pynative mode for debugging purpose.

[Pynative mode with ms_function ](https://www.mindspore.cn/tutorials/zh-CN/r1.8/advanced/pynative_graph/combine.html) is a mixed mode for comprising flexibility and efficiency in MindSpore. To apply pynative mode with ms_function for training, please run `train_with_func.py`, e.g.,

``` shell
python train_with_func.py --model=resnet50 --dataset=cifar10 --dataset_download  --epoch_size=10
```
>Note: this is an **experimental** function under improvement. It is not stable on MindSpore 1.8.1 or earlier versions.


## Tutorials
We provide the following jupyter notebook tutorials to help users learn to use MindCV.

- [Learn about configs](tutorials/learn_about_config.md)
- [Inference with a pretrained model](tutorials/inference.md)
- [Finetune a pretrained model on custom datasets](tutorials/finetune.md)
- [Customize your model]() //coming soon
- [Optimizing performance for vision transformer]() //coming soon
- [Deployment demo](tutorials/deployment.md)

## Model List

Currently, MindCV supports the model families listed below. More models with pre-trained weights are under development and will be released soon.

<details open>
<summary> Supported models </summary>

* Big Transfer ResNetV2 (BiT) - https://arxiv.org/abs/1912.11370
* ConvNeXt - https://arxiv.org/abs/2201.03545
* ConViT (Soft Convolutional Inductive Biases Vision Transformers)- https://arxiv.org/abs/2103.10697
* DenseNet - https://arxiv.org/abs/1608.06993
* DPN (Dual-Path Network) - https://arxiv.org/abs/1707.01629
* EfficientNet (MBConvNet Family) https://arxiv.org/abs/1905.11946
* EfficientNet V2 - https://arxiv.org/abs/2104.00298
* GhostNet - https://arxiv.org/abs/1911.11907
* GoogleNet - https://arxiv.org/abs/1409.4842
* Inception-V3 - https://arxiv.org/abs/1512.00567
* Inception-ResNet-V2 and Inception-V4 - https://arxiv.org/abs/1602.07261
* MNASNet - https://arxiv.org/abs/1807.11626
* MobileNet-V1 - https://arxiv.org/abs/1704.04861
* MobileNet-V2 - https://arxiv.org/abs/1801.04381
* MobileNet-V3 (MBConvNet w/ Efficient Head) - https://arxiv.org/abs/1905.02244
* NASNet - https://arxiv.org/abs/1707.07012
* PNasNet - https://arxiv.org/abs/1712.00559
* PVT (Pyramid Vision Transformer) - https://arxiv.org/abs/2102.12122
* PoolFormer models - https://github.com/sail-sg/poolformer
* RegNet - https://arxiv.org/abs/2003.13678
* RepMLP https://arxiv.org/abs/2105.01883
* RepVGG - https://arxiv.org/abs/2101.03697
* ResNet (v1b/v1.5) - https://arxiv.org/abs/1512.03385
* ResNeXt - https://arxiv.org/abs/1611.05431
* Res2Net - https://arxiv.org/abs/1904.01169
* ReXNet - https://arxiv.org/abs/2007.00992
* ShuffleNet v1 - https://arxiv.org/abs/1707.01083
* ShuffleNet v2 - https://arxiv.org/abs/1807.11164
* SKNet - https://arxiv.org/abs/1903.06586
* SqueezeNet - https://arxiv.org/abs/1602.07360
* Swin Transformer - https://arxiv.org/abs/2103.14030
* VGG - https://arxiv.org/abs/1409.1556
* Visformer - https://arxiv.org/abs/2104.12533
* Vision Transformer (ViT) - https://arxiv.org/abs/2010.11929
* Xception - https://arxiv.org/abs/1610.02357

Please see [configs](./configs) for the details about model performance and pretrained weights.

</details>

## Supported Algorithms
<details open>
<summary> Supported algorithms </summary>

* Augmentation
	* [AutoAugment](https://arxiv.org/abs/1805.09501)
	* [RandAugment](https://arxiv.org/abs/1909.13719)
	* [Repeated Augmentation](https://openaccess.thecvf.com/content_CVPR_2020/papers/Hoffer_Augment_Your_Batch_Improving_Generalization_Through_Instance_Repetition_CVPR_2020_paper.pdf)
	* RandErasing (Cutout)
	* CutMix
	* Mixup
	* RandomResizeCrop
	* Color Jitter, Flip, etc
* Optimizer
	* Adam
	* Adamw
	* [Lion](https://arxiv.org/abs/2302.06675)
	* Adan (experimental)
	* AdaGrad
	* LAMB
	* Momentum
	* RMSProp
	* SGD
	* NAdam
* LR Scheduler
	* Warmup Cosine Decay
	* Step LR
	* Polynomial Decay
	* Exponential Decay
* Regularization
	* Weight Decay
	* Label Smoothing
	* Stochastic Depth (depends on networks)
	* Dropout (depends on networks)
* Loss
	* Cross Entropy (w/ class weight and auxiliary logit support)
	* Binary Cross Entropy  (w/ class weight and auxiliary logit support)
	* Soft Cross Entropy Loss (automatically enabled if mixup or label smoothing is used)
	* Soft Binary Cross Entropy Loss (automatically enabled if mixup or label smoothing is used)
* Ensemble
	* Warmup EMA (Exponential Moving Average)
</details>

## Notes
### What is New
- 2023/04/28
1. Add some new models, listed as following
    - [VGG](configs/vgg)
    - [DPN](configs/dpn)
    - [ResNet v2](configs/resnetv2)
    - [MnasNet](configs/mnasnet)
    - [MixNet](configs/mixnet)
    - [RepVGG](configs/repvgg)
    - [ConvNeXt](configs/convnext)
    - [Swin Transformer](configs/swintransformer)
    - [EdgeNeXt](configs/edgenext)
    - [CrossViT](configs/crossvit)
    - [XCiT](configs/xcit)
    - [CoAT](configs/coat)
    - [PiT](configs/pit)
    - [PVT v2](configs/pvt_v2)
    - [MobileViT](configs/mobilevit)
2. Bug fix:
    - Setting the same random seed for each rank
    - Checking if options from yaml config exist in argument parser
    - Initializing flag variable as `Tensor` in Optimizer `Adan`

- 2023/03/25
1. Update checkpoints for pretrained ResNet for better accuracy
    - ResNet18 (from 70.09 to 70.31 @Top1 accuracy)
    - ResNet34 (from 73.69 to 74.15 @Top1 accuracy)
    - ResNet50 (from 76.64 to 76.69 @Top1 accuracy)
    - ResNet101 (from 77.63 to 78.24 @Top1 accuracy)
    - ResNet152 (from 78.63 to 78.72 @Top1 accuracy)
2. Rename checkpoint file name to follow naming rule ({model_scale-sha256sum.ckpt}) and update download URLs.

- 2023/03/05
1. Add Lion (EvoLved Sign Momentum) optimizer from paper https://arxiv.org/abs/2302.06675
	- To replace adamw with lion, LR is usually 3-10x smaller, and weight decay is usually 3-10x larger than adamw.
2. Add 6 new models with training recipes and pretrained weights for
	- [HRNet](configs/hrnet)
	- [SENet](configs/senet)
	- [GoogLeNet](configs/googlenet)
	- [Inception V3](configs/inception_v3)
	- [Inception V4](configs/inception_v4)
	- [Xception](configs/xception)
3. Support gradient clip
4. Arg name `use_ema` changed to **`ema`**, add `ema: True` in yaml to enable EMA.

- 2023/01/10
1. MindCV v0.1 released! It can be installed via PyPI `pip install mindcv` now.
2. Add training recipe and trained weights of googlenet, inception_v3, inception_v4, xception

- 2022/12/09
1. Support lr warmup for all lr scheduling algorithms besides cosine decay.
2. Add repeated augmentation, which can be enabled by setting `--aug_repeats` to be a value larger than 1 (typically, 3 or 4 is a common choice).
3. Add EMA.
4. Improve BCE loss to support mixup/cutmix.

- 2022/11/21
1. Add visualization for loss and acc curves
2. Support epochwise lr warmup cosine decay (previous is stepwise)
- 2022/11/09
1. Add 7 pretrained ViT models.
2. Add RandAugment augmentation.
3. Fix CutMix efficiency issue and CutMix and Mixup can be used together.
4. Fix lr plot and scheduling bug.
- 2022/10/12
1. Both BCE and CE loss now support class-weight config, label smoothing, and auxiliary logit input (for networks like inception).
- 2022/09/13
1. Add Adan optimizer (experimental)

### How to Contribute

We appreciate all kind of contributions including issues and PRs to make MindCV better.

Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for the contributing guideline. Please follow the [Model Template and Guideline](mindcv/models/model_template.md) for contributing a model that fits the overall interface :)

### License

This project follows the [Apache License 2.0](LICENSE.md) open-source license.

### Acknowledgement

MindCV is an open-source project jointly developed by the MindSpore team, Xidian University, and Xi'an Jiaotong University.
Sincere thanks to all participating researchers and developers for their hard work on this project.
We also acknowledge the computing resources provided by [OpenI](https://openi.pcl.ac.cn/).

### Citation

If you find this project useful in your research, please consider citing:

```latex
@misc{MindSpore Computer Vision 2022,
    title={{MindSpore Computer  Vision}:MindSpore Computer Vision Toolbox and Benchmark},
    author={MindSpore Vision Contributors},
    howpublished = {\url{https://github.com/mindspore-lab/mindcv/}},
    year={2022}
}
```
