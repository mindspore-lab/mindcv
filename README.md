<div align="center" markdown>

# MindCV

[![CI](https://github.com/mindspore-lab/mindcv/actions/workflows/ci.yml/badge.svg)](https://github.com/mindspore-lab/mindcv/actions/workflows/ci.yml)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mindcv)](https://pypi.org/project/mindcv)
[![PyPI](https://img.shields.io/pypi/v/mindcv)](https://pypi.org/project/mindcv)
[![docs](https://github.com/mindspore-lab/mindcv/actions/workflows/docs.yml/badge.svg)](https://mindspore-lab.github.io/mindcv)
[![license](https://img.shields.io/github/license/mindspore-lab/mindcv.svg)](https://github.com/mindspore-lab/mindcv/blob/main/LICENSE.md)
[![open issues](https://img.shields.io/github/issues/mindspore-lab/mindcv)](https://github.com/mindspore-lab/mindcv/issues)
[![PRs](https://img.shields.io/badge/PRs-welcome-pink.svg)](https://github.com/mindspore-lab/mindcv/pulls)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

[📝Documentation](https://mindspore-lab.github.io/mindcv/) |
[🚀Installation](https://mindspore-lab.github.io/mindcv/installation/) |
[🎁Model Zoo](https://mindspore-lab.github.io/mindcv/modelzoo/) |
[🎉Update News](https://github.com/mindspore-lab/mindcv/blob/main/RELEASE.md) |
[🐛Reporting Issues](https://github.com/mindspore-lab/mindcv/issues/new/choose)

English | [中文](README_CN.md)

</div>

## Introduction

MindCV is an open-source toolbox for computer vision research and development based on [MindSpore](https://www.mindspore.cn/en). It collects a series of classic and SoTA vision models, such as ResNet and SwinTransformer, along with their pre-trained weights and training strategies. SoTA methods such as auto augmentation are also provided for performance improvement. With the decoupled module design, it is easy to apply or adapt MindCV to your own CV tasks.

The following is the corresponding `mindcv` versions and supported `mindspore` versions.

| mindcv |  mindspore  |
| :----: | :---------: |
|  main  |   master    |
| v0.4.0 | 2.3.0/2.3.1 |
| 0.3.0  |   2.2.10    |
|  0.2   |     2.0     |
|  0.1   |     1.8     |


### Major Features

- **Easy-to-Use.** MindCV decomposes the vision framework into various configurable components. It is easy to customize your data pipeline, models, and learning pipeline with MindCV:

    ```pycon
    >>> import mindcv
    # create a dataset
    >>> dataset = mindcv.create_dataset('cifar10', download=True)
    # create a model
    >>> network = mindcv.create_model('resnet50', pretrained=True)
    ```

    Users can customize and launch their transfer learning or training task in one command line.

    ```shell
    # transfer learning in one command line
    python train.py --model=swin_tiny --pretrained --opt=adamw --lr=0.001 --data_dir=/path/to/data
    ```

- **State-of-The-Art.** MindCV provides various CNN-based and Transformer-based vision models including SwinTransformer. Their pretrained weights and performance reports are provided to help users select and reuse the right model:

- **Flexibility and efficiency.** MindCV is built on MindSpore which is an efficient DL framework that can be run on different hardware platforms (GPU/CPU/Ascend). It supports both graph mode for high efficiency and pynative mode for flexibility.

## Model Zoo

The performance of the models trained with MindCV is summarized in [here](https://mindspore-lab.github.io/mindcv/modelzoo/), where the training recipes and weights are both available.

Model introduction and training details can be viewed in each sub-folder under [configs](configs).

## Installation

See [Installation](https://mindspore-lab.github.io/mindcv/installation/) for details.

## Getting Started

### Hands-on Tutorial

To get started with MindCV, please see the [Quick Start](docs/en/tutorials/quick_start.md), which will give you a quick tour on each key component and the train/validate/predict pipelines.

Below are a few code snippets for your taste.

```pycon
>>> import mindcv
# List and find a pretrained vision model
>>> mindcv.list_models("swin*", pretrained=True)
['swin_tiny']
# Create the model object
>>> network = mindcv.create_model('swin_tiny', pretrained=True)
```
```shell
# Validate its accuracy
python validate.py --model=swin_tiny --pretrained --dataset=imagenet --val_split=validation
# {'Top_1_Accuracy': 0.80824, 'Top_5_Accuracy': 0.94802, 'loss': 1.7331367141008378}
```

**Image classification demo**

Right click on the image below and save as `dog.jpg`.

<p align="left">
  <img src="https://user-images.githubusercontent.com/8156835/210049681-89f68b9f-eb44-44e2-b689-4d30c93c6191.jpg" width=360 />
</p>

Classify the downloaded image with a pretrained SoTA model:

```shell
python infer.py --model=swin_tiny --image_path='./dog.jpg'
# {'Labrador retriever': 0.5700152, 'golden retriever': 0.034551315, 'kelpie': 0.010108651, 'Chesapeake Bay retriever': 0.008229004, 'Walker hound, Walker foxhound': 0.007791956}
```
The top-1 prediction result is labrador retriever, which is the breed of this cut dog.

### Training

It is easy to train your model on a standard or customized dataset using `train.py`, where the training strategy (e.g., augmentation, LR scheduling) can be configured with external arguments or a yaml config file.

- Standalone Training

    ```shell
    # standalone training
    python train.py --model=resnet50 --dataset=cifar10 --dataset_download
    ```

    Above is an example for training ResNet50 on CIFAR10 dataset on a CPU/GPU/Ascend device

- Distributed Training

    For large datasets like ImageNet, it is necessary to do training in distributed mode on multiple devices. This can be achieved with `msrun` and parallel features supported by MindSpore.

    ```shell
    # distributed training
    # assume you have 4 GPUs/NPUs
    msrun --bind_core=True --worker_num 4 python train.py --distribute \
        --model=densenet121 --dataset=imagenet --data_dir=/path/to/imagenet
    ```

    Notice that if you are using msrun startup with 2 devices, please add `--bind_core=True` to improve performance. For example:

    ```shell
    msrun --bind_core=True --worker_num=2--local_worker_num=2 --master_port=8118 \
    --log_dir=msrun_log --join=True --cluster_time_out=300 \
    python train.py --distribute --model=densenet121 --dataset=imagenet --data_dir=/path/to/imagenet
    ```

   > For more information, please refer to https://www.mindspore.cn/tutorials/experts/en/r2.3.1/parallel/startup_method.html

    Detailed parameter definitions can be seen in `config.py` and checked by running `python train.py --help'.

    To resume training, please set the `--ckpt_path` and `--ckpt_save_dir` arguments. The optimizer state including the learning rate of the last stopped epoch will also be recovered.

- Config and Training Strategy

    You can configure your model and other components either by specifying external parameters or by writing a yaml config file. Here is an example of training using a preset yaml file.

    ```shell
    msrun --bind_core=True --worker_num 4 python train.py -c configs/squeezenet/squeezenet_1.0_gpu.yaml
    ```

    **Pre-defined Training Strategies:**
    We provide more than 20 training recipes that achieve SoTA results on ImageNet currently.
    Please look into the [`configs`](configs) folder for details.
    Please feel free to adapt these training strategies to your own model for performance improvement, which can be easily done by modifying the yaml file.

- Train on ModelArts/OpenI Platform

    To run training on the [ModelArts](https://www.huaweicloud.com/intl/en-us/product/modelarts.html) or [OpenI](https://openi.pcl.ac.cn/) cloud platform:

    ```text
    1. Create a new training task on the cloud platform.
    2. Add run parameter `config` and specify the path to the yaml config file on the website UI interface.
    3. Add run parameter `enable_modelarts` and set True on the website UI interface.
    4. Fill in other blanks on the website and launch the training task.
    ```

**Graph Mode and PyNative Mode**:

By default, the training pipeline `train.py` is run in [graph mode](https://www.mindspore.cn/tutorials/zh-CN/r1.8/advanced/pynative_graph/mode.html#%E9%9D%99%E6%80%81%E5%9B%BE) on MindSpore, which is optimized for efficiency and parallel computing with a compiled static graph.
In contrast, [pynative mode](https://www.mindspore.cn/tutorials/zh-CN/r1.8/advanced/pynative_graph/mode.html#%E5%8A%A8%E6%80%81%E5%9B%BE) is optimized for flexibility and easy debugging. You may alter the parameter `--mode` to switch to pure pynative mode for debugging purpose.

**Mixed Mode**:

[PyNative mode with mindspore.jit](https://www.mindspore.cn/tutorials/zh-CN/r1.8/advanced/pynative_graph/combine.html) is a mixed mode for comprising flexibility and efficiency in MindSpore. To apply pynative mode with mindspore.jit for training, please run `train_with_func.py`, e.g.,

```shell
python train_with_func.py --model=resnet50 --dataset=cifar10 --dataset_download  --epoch_size=10
```

> Note: this is an **experimental** function under improvement. It is not stable on MindSpore 1.8.1 or earlier versions.

### Validation

To evaluate the model performance, please run `validate.py`

```shell
# validate a trained checkpoint
python validate.py --model=resnet50 --dataset=imagenet --data_dir=/path/to/data --ckpt_path=/path/to/model.ckpt
```

**Validation while Training**

You can also track the validation accuracy during training by enabling the `--val_while_train` option.

```shell
python train.py --model=resnet50 --dataset=cifar10 \
    --val_while_train --val_split=test --val_interval=1
```

The training loss and validation accuracy for each epoch will be saved in `{ckpt_save_dir}/results.log`.

More examples about training and validation can be seen in [examples](examples/scripts).

## Tutorials

We provide the following jupyter notebook tutorials to help users learn to use MindCV.

- [Learn about configs](docs/en/tutorials/configuration.md)
- [Inference with a pretrained model](docs/en/tutorials/inference.md)
- [Finetune a pretrained model on custom datasets](docs/en/tutorials/finetune.md)
- [Customize your model]() //coming soon
- [Optimizing performance for vision transformer]() //coming soon

## Model List

Currently, MindCV supports the model families listed below. More models with pre-trained weights are under development and will be released soon.

<details open markdown>
<summary> Supported models </summary>

* Big Transfer ResNetV2 (BiT) - https://arxiv.org/abs/1912.11370
* ConvNeXt - https://arxiv.org/abs/2201.03545
* ConViT (Soft Convolutional Inductive Biases Vision Transformers)- https://arxiv.org/abs/2103.10697
* DenseNet - https://arxiv.org/abs/1608.06993
* DPN (Dual-Path Network) - https://arxiv.org/abs/1707.01629
* EfficientNet (MBConvNet Family) https://arxiv.org/abs/1905.11946
* EfficientNet V2 - https://arxiv.org/abs/2104.00298
* GhostNet - https://arxiv.org/abs/1911.11907
* GoogLeNet - https://arxiv.org/abs/1409.4842
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

<details open markdown>
<summary> Supported algorithms </summary>

* Augmentation
    * [AutoAugment](https://arxiv.org/abs/1805.09501)
    * [RandAugment](https://arxiv.org/abs/1909.13719)
    * [Repeated Augmentation](https://openaccess.thecvf.com/content_CVPR_2020/papers/Hoffer_Augment_Your_Batch_Improving_Generalization_Through_Instance_Repetition_CVPR_2020_paper.pdf)
    * RandErasing (Cutout)
    * CutMix
    * MixUp
    * RandomResizeCrop
    * Color Jitter, Flip, etc
* Optimizer
    * Adam
    * AdamW
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

## What is New

- 2024/1/17

Release `0.3.0` is published. We will drop MindSpore 1.x in the future release.

1. New models:
   - Y-16GF of [RegNet](configs/regnet)
   - [SwinTransformerV2](configs/swintransformerv2)
   - [VOLO](configs/volo)
   - [CMT](configs/cmt)
   - [HaloNet](configs/halonet)
   - [SSD](examples/det/ssd)
   - [DeepLabV3](examples/seg/deeplabv3)
   - [CLIP](examples/clip) & [OpenCLIP](examples/open_clip)
2. Features:
   - AsymmetricLoss & JSDCrossEntropy
   - Augmentations Split
   - Customized AMP
3. Bug fixes:
   - Since the classifier weights are not fully deleted, you may encounter an error passing in the `num_classes` when creating a pre-trained model.
4. Refactoring:
   - The names of many models have been refactored for better understanding.
   - [Script](mindcv/models/vit.py) of `VisionTransformer`.
   - [Script](train_with_func.py) of Mixed(PyNative+jit) mode training.
5. Documentation:
   - A guide of how to extract multiscale features from backbone.
   - A guide of how to finetune the pre-trained model on a custom dataset.
6. BREAKING CHANGES:
   - We are going to drop support of MindSpore 1.x for it's EOL.
   - Configuration `filter_bias_and_bn` will be removed and renamed as `weight_decay_filter`,
   due to a prolonged misunderstanding of the MindSpore optimizer.
   We will migrate the existing training recipes, but the signature change of function `create_optimizer` will be incompatible
   and the old version training recipes will also be incompatible. See [PR/752](https://github.com/mindspore-lab/mindcv/pull/752) for details.

See [RELEASE](RELEASE.md) for detailed history.

## How to Contribute

We appreciate all kinds of contributions including issues and PRs to make MindCV better.

Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for the contributing guideline.
Please follow the [Model Template and Guideline](docs/en/how_to_guides/write_a_new_model.md) for contributing a model that fits the overall interface :)

## License

This project follows the [Apache License 2.0](LICENSE.md) open-source license.

## Acknowledgement

MindCV is an open-source project jointly developed by the MindSpore team, Xidian University, and Xi'an Jiaotong University.
Sincere thanks to all participating researchers and developers for their hard work on this project.
We also acknowledge the computing resources provided by [OpenI](https://openi.pcl.ac.cn/).

## Citation

If you find this project useful in your research, please consider citing:

```latex
@misc{MindSpore Computer Vision 2022,
    title={{MindSpore Computer Vision}:MindSpore Computer Vision Toolbox and Benchmark},
    author={MindSpore Vision Contributors},
    howpublished = {\url{https://github.com/mindspore-lab/mindcv/}},
    year={2022}
}
```
