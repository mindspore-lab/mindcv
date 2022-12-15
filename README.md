# MindCV

<p align="left">
    <a href="https://mindcv-ai.readthedocs.io/en/latest">
        <img alt="docs" src="https://img.shields.io/badge/docs-latest-blue">
    </a>
    <a href="https://github.com/mindspore-lab/mindcv/blob/main/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/mindspore-lab/mindcv.svg">
    </a>
    <a href="https://github.com/mindspore-lab/mindcv/pulls">
        <img alt="PRs Welcome" src="https://img.shields.io/badge/PRs-welcome-pink.svg">
    </a>
    <a href="https://github.com/mindspore-lab/mindcv/issues">
        <img alt="open issues" src="https://img.shields.io/github/issues/mindspore-lab/mindcv">
    </a>
    <!---
    <a href="https://github.com/mindspore-lab/mindcv/tags">
        <img alt="GitHub tags" src="https://img.shields.io/github/tags/mindspore-lab/mindcv">
    </a>
    -->
</p>


| **Build Type**   |`Linux`           |`MacOS`           |`Windows`         |
| :---:            | :---:            | :---:            | :---:            |
| **Build Status** | [![Status](https://github.com/mindspore-lab/mindcv/actions/workflows/main.yml/badge.svg)](https://github.com/mindspore-lab/mindcv/actions) | [![Status](https://github.com/mindspore-lab/mindcv/actions/workflows/mac.yml/badge.svg)](https://github.com/mindspore-lab/mindcv/actions) | Not tested|

[Introduction](#introduction) |
[Installation](#installation) |
[Get Started](#get-started) |
[Tutorials](#tutorials) |
[Model List](#model-list) |
[Supported Algorithms](#supported-algorithms) |
[Notes](#notes) 


## Introduction
MindCV is an open source toolbox for computer vision research and development based on [MindSpore](https://www.mindspore.cn/en). It collects a series of classic and SoTA vision models, such as ResNet and SwinTransformer, along with their pretrained weights. SoTA methods such as AutoAugment are also provided for performance improvement. With the decoupled module design, it is easy to apply or adapt MindCV to your own CV tasks. 

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

- **Flexibility and efficiency.** MindCV is bulit on MindSpore which is an efficent DL framework that can run on different hardward platforms (GPU/CPU/Ascend). It supports both graph mode for high efficiency and pynative mode for flexibity.
	
</details>
	
### Benchmark Results

Coming soon.


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

MindSpore can be easily installed by following the official [instruction](https://www.mindspore.cn/install) where you can select your hardware platform for the best fit. To run in distributed mode, [openmpi](https://www.open-mpi.org/software/ompi/v4.0/) is required to install.   

The following instructions assume the desired dependency is fulfilled. 
<!---
### Install with pip
MindCV can be installed with pip. 
```shell
pip install https://github.com/mindspore-lab/mindcv/releases/download/v0.0.1-beta/mindcv-0.0.1b0-py3-none-any.whl
```
-->

### Install from source
To install MindCV from source, please run:
```shell
pip install git+https://github.com/mindspore-lab/mindcv.git
```

#### Notes: 
* MindCV can be installed on Linux and Mac but not on Windows currently.

## Get Started 

### Hands-on Tutorial

To get started with MindCV, please see the [transfer learning tutorial](tutorials/finetune.ipynb), which will give a quick tour on each key component and the train/validate/predict pipelines in MindCV. 

Below is a few code snippets for your taste. 

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
  <img src="./tutorials/dog.jpg" width=360 />
</p>

Infer the input image with a pretrained SoTA model,

```python
>>> !python infer.py --model=swin_tiny --image_path='./tutorials/dog.jpg'
{'Labrador retriever': 0.5700152, 'golden retriever': 0.034551315, 'kelpie': 0.010108651, 'Chesapeake Bay retriever': 0.008229004, 'Walker hound, Walker foxhound': 0.007791956}
```
The top-1 prediction result is labrador retriever (拉布拉多犬), which is the breed of this cut dog.

### Useful Script Guidelines
It is easy to train your model on standard datasets or your own dataset with MindCV. Model training, transfer learning, or evaluaiton can be done using one or a few line of code with flexible configuration. 

- Standalone Training

It is easy to do model training with `train.py`. Here is an example for training a DenseNet on CIFAR10 dataset using one computing device (i.e., standalone GPU).
``` shell
python train.py --model=resnet50 --dataset=cifar10 --dataset_download
```

For more parameter description, please run `python train.py --help'. You can define change model, optimizer, and other hyper-parameters easily.

**Validation while training.** To track the validation accuracy change during traing, please enable `--val_while_train`, for example

```python
python train.py --model=resnet50 --dataset=cifar10 \
		--val_while_train --val_split=test --val_interval=1
``` 

The training loss and validation accuracy for each epoch  will be saved in `{ckpt_save_dir}/results.log`.

**Resume training.** To resume training, please specify `--ckpt_path` for the checkpoint where you want to resume and `--ckpt_save_dir`. The optimizer state including learning rate of the last epoch will also be recovered. 

```python
python train.py --model=resnet50 --dataset=cifar10 \
		--ckpt_save_dir=checkpoints --ckpt_path=checkpoints/resnet50_30-100.ckpt
``` 

- Distributed Training

For large datasets like ImageNet, it is necessary to do training in distributed mode on multiple devices, which is well supported in MindCV. The following script is an example for training DenseNet121 on ImageNet with 4 GPUs.   

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3  # suppose there are 4 GPUs
mpirun --allow-run-as-root -n 4 python train.py --distribute \
	--model=densenet121 --dataset=imagenet --data_dir=./datasets/imagenet   
```

- Configuration with Yaml

You can configure your model and other components either by specifying external parameters or by using a yaml config file. Here is an example for training using a preset yaml file.

```shell
mpirun --allow-run-as-root -n 4 python train.py -c configs/squeezenet/squeezenet_1.0_gpu.yaml    
```

More [yaml config files](configs) used to yield competitive results on ImageNet training can be found in the `configs` folder. 


- Validation

It is easy to validate a trained model with `validate.py`. 
```python
# validate a trained checkpoint
python validate.py --model=resnet50 --dataset=imagenet --val_split=validation --ckpt_path='./ckpt/densenet121-best.ckpt' 
``` 

- Pynative mode with ms_function (Advanced)

By default, the training pipeline (`train.py`) is run in [graph mode](https://www.mindspore.cn/tutorials/zh-CN/r1.8/advanced/pynative_graph/mode.html), which is optimized for efficienty and speed but may not be flexible enough for debugging. You may alter the parameter `--mode` to switch to pure pynative mode for debugging purpose.

[Pynative mode with ms_function ](https://www.mindspore.cn/tutorials/zh-CN/r1.8/advanced/pynative_graph/combine.html) is a mixed mode for comprising flexibity and efficiency in MindSpore. To switch to pynative mode with ms_function, please use `train_with_func.py` instead, for example:

``` shell
python train_with_func.py --model=resnet50 --dataset=cifar10 --dataset_download  --epoch_size=10  
```

For more examples, see [examples/scripts](examples/scripts). 

## Tutorials
We provide [jupyter notebook tutorials](tutorials) for  

- [Learn about configs](tutorials/learn_about_config.ipynb)  
- [Inference with a pretrained model](tutorials/inference.ipynb) 
- [Finetune a pretrained model on custom datasets](tutorials/finetune.ipynb) 
- [Customize models] //coming soon
- [Optimizing performance for vision transformer] //coming soon
- [Deployment demo](tutorials/deployment.ipynb) 

## Model List

Currently, MindCV supports the model families listed below. More models with pretrained weights are under development and will be released soon.

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
* PoolFormer models w/ weights adapted from https://github.com/sail-sg/poolformer
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
	* Cross Entropy (w/ class weight and auxilary logit support)
	* Binary Cross Entropy  (w/ class weight and auxilary logit support)
	* Soft Cross Entropy Loss (automatically enabled if mixup or label smoothing is used)
	* Soft Binary Cross Entropy Loss (automatically enabled if mixup or label smoothing is used)
* Ensemble
	* Warmup EMA (Exponential Moving Average)
</details>

## Notes
### What is New 
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
1. Both BCE and CE loss now support class-weight config, label smoothing, and auxilary logit input (for networks like inception).
- 2022/09/13
1. Add Adan optimizer (experimental) 

### How to Contribute

We appreciate all contributions including issues and PRs to make MindCV better. 

Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for the contributing guideline. Please follow the [Model Template and Guideline](mindcv/models/model-template.md) for contributing a model that fits the overall interface :)

### License

This project is released under the [Apache License 2.0](LICENSE.md).

### Acknowledgement

MindCV is an open source project that welcome any contribution and feedback. We wish that the toolbox and benchmark could serve the growing research community by providing a flexible as well as standardized toolkit to reimplement existing methods and develop their own new computer vision methods.

### Citation

If you find this project useful in your research, please consider citing:

```latex
@misc{MindSpore Computer Vision 2022,
    title={{MindSpore Computer  Vision}:MindSpore Computer Vision Toolbox and Benchmark},
    author={MindSpore Vision Contributors},
    howpublished = {\url{https://github.com/mindlab-ecosystem/mindcv/}},
    year={2022}
}
```
