---
hide:
  - navigation
---

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

</div>

## Introduction

MindCV is an open-source toolbox for computer vision research and development based on [MindSpore](https://www.mindspore.cn/en). It collects a series of classic and SoTA vision models, such as ResNet and SwinTransformer, along with their pre-trained weights and training strategies. SoTA methods such as auto augmentation are also provided for performance improvement. With the decoupled module design, it is easy to apply or adapt MindCV to your own CV tasks.

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

The performance of the models trained with MindCV is summarized in [here](./modelzoo.md), where the training recipes and weights are both available.

Model introduction and training details can be viewed in each sub-folder under [configs](https://github.com/mindspore-lab/mindcv/tree/main/configs).

## Installation

See [Installation](./installation.md) for details.

## Getting Started

### Hands-on Tutorial

To get started with MindCV, please see the [Quick Start](./tutorials/quick_start.md), which will give you a quick tour of each key component and the train/validate/predict pipelines.

Below are a few code snippets for your taste.

```pycon
>>> import mindcv
# List and find a pretrained vision model
>>> mindcv.list_models("swin*", pretrained=True)
['swin_tiny']
# Create the model object
>>> network = mindcv.create_model('swin_tiny', pretrained=True)
# Validate its accuracy
>>> !python validate.py --model=swin_tiny --pretrained --dataset=imagenet --val_split=validation
{'Top_1_Accuracy': 0.80824, 'Top_5_Accuracy': 0.94802, 'loss': 1.7331367141008378}
```

???+ example "Image Classification Demo"

    Right click on the image below and save as `dog.jpg`.

    <p align="center">
      <img src="https://user-images.githubusercontent.com/8156835/210049681-89f68b9f-eb44-44e2-b689-4d30c93c6191.jpg" width=360 />
    </p>

    Classify the downloaded image with a pretrained SoTA model:

    ```pycon
    >>> !python infer.py --model=swin_tiny --image_path='./dog.jpg'
    {'Labrador retriever': 0.5700152, 'golden retriever': 0.034551315, 'kelpie': 0.010108651, 'Chesapeake Bay retriever': 0.008229004, 'Walker hound, Walker foxhound': 0.007791956}
    ```

    The top-1 prediction result is labrador retriever, which is the breed of this cut dog.

### Training

It is easy to train your model on a standard or customized dataset using `train.py`, where the training strategy (e.g., augmentation, LR scheduling) can be configured with external arguments or a yaml config file.

- Standalone Training

    ```shell
    # standalone training
    python train.py --model=resnet50 --dataset=cifar10 --dataset_download
    ```

    Above is an example of training ResNet50 on CIFAR10 dataset on a CPU/GPU/Ascend device

- Distributed Training

    For large datasets like ImageNet, it is necessary to do training in distributed mode on multiple devices. This can be achieved with `mpirun` and parallel features supported by MindSpore.

    ```shell
    # distributed training
    # assume you have 4 GPUs/NPUs
    mpirun -n 4 python train.py --distribute \
        --model=densenet121 --dataset=imagenet --data_dir=/path/to/imagenet
    ```
    > Notes: If the script is executed by the root user, the `--allow-run-as-root` parameter must be added to `mpirun`.

    Detailed parameter definitions can be seen in `config.py` and checked by running `python train.py --help'.

    To resume training, please set the `--ckpt_path` and `--ckpt_save_dir` arguments. The optimizer state including the learning rate of the last stopped epoch will also be recovered.

- Config and Training Strategy

    You can configure your model and other components either by specifying external parameters or by writing a yaml config file. Here is an example of training using a preset yaml file.

    ```shell
    mpirun --allow-run-as-root -n 4 python train.py -c configs/squeezenet/squeezenet_1.0_gpu.yaml
    ```

    !!! tip "Pre-defined Training Strategies"
        We provide more than 20 training recipes that achieve SoTA results on ImageNet currently.
        Please look into the [`configs`](https://github.com/mindspore-lab/mindcv/tree/main/configs) folder for details.
        Please feel free to adapt these training strategies to your own model for performance improvement, which can be easily done by modifying the yaml file.

- Train on ModelArts/OpenI Platform

    To run training on the [ModelArts](https://www.huaweicloud.com/intl/en-us/product/modelarts.html) or [OpenI](https://openi.pcl.ac.cn/) cloud platform:

    ```text
    1. Create a new training task on the cloud platform.
    2. Add the parameter `config` and specify the path to the yaml config file on the website UI interface.
    3. Add the parameter `enable_modelarts` and set True on the website UI interface.
    4. Fill in other blanks on the website and launch the training task.
    ```

!!! tip "Graph Mode and PyNative Mode"

    By default, the training pipeline `train.py` is run in [graph mode](https://www.mindspore.cn/tutorials/zh-CN/r1.8/advanced/pynative_graph/mode.html#%E9%9D%99%E6%80%81%E5%9B%BE) on MindSpore, which is optimized for efficiency and parallel computing with a compiled static graph.
    In contrast, [pynative mode](https://www.mindspore.cn/tutorials/zh-CN/r1.8/advanced/pynative_graph/mode.html#%E5%8A%A8%E6%80%81%E5%9B%BE) is optimized for flexibility and easy debugging. You may alter the parameter `--mode` to switch to pure pynative mode for debugging purpose.

!!! warning "Mixed Mode"

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

!!! tip "Validation while Training"

    You can also track the validation accuracy during training by enabling the `--val_while_train` option.

    ```shell
    python train.py --model=resnet50 --dataset=cifar10 \
        --val_while_train --val_split=test --val_interval=1
    ```

    The training loss and validation accuracy for each epoch will be saved in `${ckpt_save_dir}/results.log`.

    More examples about training and validation can be seen in [examples](https://github.com/mindspore-lab/mindcv/tree/main/examples).

## Tutorials

We provide the following jupyter notebook tutorials to help users learn to use MindCV.

- [Learn about configs](./tutorials/configuration.md)
- [Inference with a pretrained model](./tutorials/inference.md)
- [Finetune a pretrained model on custom datasets](./tutorials/finetune.md)
- [Customize your model]() //coming soon
- [Optimizing performance for vision transformer]() //coming soon
- [Deployment demo](./tutorials/deployment.md)

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

## How to Contribute

We appreciate all kinds of contributions including issues and PRs to make MindCV better.

Please refer to [CONTRIBUTING](./notes/contributing.md) for the contributing guideline.
Please follow the [Model Template and Guideline](./how_to_guides/write_a_new_model.md) for contributing a model that fits the overall interface :)

## License

This project follows the [Apache License 2.0](https://github.com/mindspore-lab/mindcv/blob/main/LICENSE.md) open-source license.

## Acknowledgement

MindCV is an open-source project jointly developed by the MindSpore team, Xidian University, and Xi'an Jiaotong University.
Sincere thanks to all participating researchers and developers for their hard work on this project.
We also acknowledge the computing resources provided by [OpenI](https://openi.pcl.ac.cn/).

## Citation

If you find this project useful in your research, please consider citing:

```latex
@misc{MindSpore Computer Vision 2022,
    title={{MindSpore Computer  Vision}:MindSpore Computer Vision Toolbox and Benchmark},
    author={MindSpore Vision Contributors},
    howpublished = {\url{https://github.com/mindspore-lab/mindcv/}},
    year={2022}
}
```
