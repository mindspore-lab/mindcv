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

[English](README.md) | 中文

[简介](#简介) |
[安装](#安装) |
[快速入门](#快速入门) |
[教程](#教程) |
[模型列表](#模型列表) |
[支持算法](#支持算法) |
[日志](#日志)

</div>

## 简介

MindCV是一个基于 [MindSpore](https://www.mindspore.cn/)
开发的，致力于计算机视觉相关技术研发的开源工具箱。它提供大量的计算机视觉领域的经典模型和SoTA模型以及它们的预训练权重和训练策略。同时，还提供了自动增强等SoTA算法来提高模型性能。通过解耦的模块设计，您可以轻松地将MindCV应用到您自己的CV任务中。

<details open>
<summary> 主要特性 </summary>

- **高易用性** MindCV将视觉任务分解为各种可配置的组件，用户可以轻松地构建自己的数据处理和模型训练流程。

```python
>>> import mindcv
# 创建数据集
>>> dataset = mindcv.create_dataset('cifar10', download=True)
# 创建模型
>>> network = mindcv.create_model('resnet50', pretrained=True)
```

用户可通过预定义的训练和微调脚本，快速配置并完成训练或迁移学习任务。

```shell
# 配置和启动迁移学习任务
python train.py --model swin_tiny --pretrained --opt=adamw --lr=0.001 --data_dir=/path/to/dataset
```

- **高性能** MindCV集成了大量基于CNN和和Transformer的高性能模型, 如SwinTransformer，并提供预训练权重、训练策略和性能报告，帮助用户快速选型并将其应用于视觉模型。

- **灵活高效** MindCV基于高效的深度学习框架MindSpore开发，具有自动并行和自动微分等特性，支持不同硬件平台上（CPU/GPU/Ascend），同时支持效率优化的静态图模式和调试灵活的动态图模式。

</details>

### 性能结果


基于MindCV进行模型实现和重训练的汇总结果详见[benchmark_results.md](./benchmark_results.md), 所用到的训练策略和训练后的模型权重均可通过表中链接获取。

各模型讲解和训练说明详见[configs](configs)目录。


## 安装

### 依赖

- mindspore >= 1.8.1
- numpy >= 1.17.0
- pyyaml >= 5.3
- tqdm
- openmpi 4.0.3 (分布式模式需要使用)

运行以下脚本，安装相关依赖。

```shell
pip install -r requirements.txt
```

用户可遵从[官方指导](https://www.mindspore.cn/install) 并根据自身使用的硬件平台选择最适合您的MindSpore版本来进行安装。如果需要在分布式条件下使用，还需安装[openmpi](https://www.open-mpi.org/software/ompi/v4.0/) 。

之后的说明将默认用户已正确安装好相关依赖。

### PyPI安装

MindCV的已发布版本可以通过PyPI安装。

```shell
pip install mindcv
```

### 源码安装

Git上最新的MindCV可以通过以下指令安装。

```shell
pip install git+https://github.com/mindspore-lab/mindcv.git
```

> 注：MindCV可以在Linux和Mac系统安装，但是目前还不能在Windows系统上安装。

## 快速入门

### 上手教程

在开始上手MindCV前，可以阅读MindCV的[迁移学习教程](tutorials/finetune_CN.md)，该教程可以帮助用户快速了解MindCV的各个重要组件以及训练、验证、测试流程。

以下是一些供您快速体验的代码样例。

```python
>>> import mindcv
# 列出满足条件的预训练模型名称
>>> mindcv.list_models("swin*", pretrained=True)
['swin_tiny']
# 创建模型
>>> network = mindcv.create_model('swin_tiny', pretrained=True)
# 验证模型的准确率
>>> !python validate.py - -model = swin_tiny - -pretrained - -dataset = imagenet - -val_split = validation
{'Top_1_Accuracy': 0.808343989769821, 'Top_5_Accuracy': 0.9527253836317136, 'loss': 0.8474242982580839}
```

**图片分类示例**

<p align="left">
  <img src="https://user-images.githubusercontent.com/8156835/210049681-89f68b9f-eb44-44e2-b689-4d30c93c6191.jpg" width=360 />
</p>

使用加载了预训练参数的SoTA模型对一张图片进行推理。

```python
>>> !python infer.py - -model = swin_tiny - -image_path = './tutorials/data/test/dog/dog.jpg'
{'Labrador retriever': 0.5700152, 'golden retriever': 0.034551315, 'kelpie': 0.010108651,
 'Chesapeake Bay retriever': 0.008229004, 'Walker hound, Walker foxhound': 0.007791956}
```

预测结果排名前1的是拉布拉多犬，正是这张图片里的狗狗的品种。

### 模型训练

通过`train.py`，用户可以很容易地在标准数据集或自定义数据集上训练模型，用户可以通过外部变量或者yaml配文件来设置训练策略（如数据增强、学习路策略）。

- 单卡训练

```shell
# 单卡训练
python train.py --model resnet50 --dataset cifar10 --dataset_download
```

以上代码是在CIFAR10数据集上单卡（CPU/GPU/Ascend）训练ResNet的示例，通过`model`和`dataset`参数分别指定需要训练的模型和数据集。

- 分布式训练

对于像ImageNet这样的大型数据集，有必要在多个设备上以分布式模式进行训练。基于MindSpore对分布式相关功能的良好支持，用户可以使用`mpirun`来进行模型的分布式训练。

```shell
# 分布式训练
# 假设你有4张GPU或者NPU卡
mpirun --allow-run-as-root -n 4 python train.py --distribute \
	--model densenet121 --dataset imagenet --data_dir ./datasets/imagenet
```

完整的参数列表及说明在`config.py`中定义，可运行`python train.py --help`快速查看。

如需恢复训练，请指定`--ckpt_path`和`--ckpt_save_dir`参数，脚本将加载路径中的模型权重和优化器状态，并恢复中断的训练进程。


- 超参配置和预训练策略


您可以编写yaml文件或设置外部参数来指定配置数据、模型、优化器等组件及其超参。以下是使用预设的训练策略（yaml文件）进行模型训练的示例。

```shell
mpirun --allow-run-as-root -n 4 python train.py -c configs/squeezenet/squeezenet_1.0_gpu.yaml
```

**预定义的训练策略** MindCV目前提前了超过20种模型训练策略，在ImageNet取得SoTA性能。具体的参数配置和详细精度性能汇总请见[`configs`](configs)文件夹。您可以便捷将这些训练策略用于您的模型训练中以提高性能（复用或修改相应的yaml文件即可）


- 在ModelArts/OpenI平台上训练

在[ModelArts](https://www.huaweicloud.com/intl/en-us/product/modelarts.html)或[OpenI](https://openi.pcl.ac.cn/)云平台上进行训练，需要执行以下操作，：

```
1、在云平台上创建新的训练任务。
2、在网站UI界面添加运行参数`config`，并指定yaml配置文件的路径。
3、在网站UI界面添加运行参数`enable_modelarts`并设置为True。
4、在网站上填写其他训练信息并启动培训任务。
```

### 模型验证

使用`validate.py`可以便捷地验证训练好的模型。

```shell
# 验证模型
python validate.py --model=resnet50 --dataset=imagenet --data_dir=/path/to/data --ckpt_path=/path/to/model.ckpt
```

- 训练过程中进行验证

当需要在训练过程中，跟踪模型在测试集上精度的变化时，请启用参数`--val_while_train`，如下

```shell
python train.py --model=resnet50 --dataset=cifar10 \
		--val_while_train --val_split=test --val_interval=1
```

各轮次的训练损失和测试精度将保存在`{ckpt_save_dir}/results.log`中。


- 静态图和动态图模式

在默认情况下，模型训练（`train.py`）在MindSpore上以[图模式](https://www.mindspore.cn/tutorials/zh-CN/r1.8/advanced/pynative_graph/mode.html) 运行，该模式对使用静态图编译对性能和并行计算进行了优化。相比之下，[pynative模式](https://www.mindspore.cn/tutorials/zh-CN/r1.8/advanced/pynative_graph/mode.html#%E5%8A%A8%E6%80%81%E5%9B%BE)的优势在于灵活性和易于调试。
为了方便调试，您可以将参数`--mode`设为1以将运行模式设置为调试模式。

[基于ms_function的混合模式](https://www.mindspore.cn/tutorials/zh-CN/r1.8/advanced/pynative_graph/combine.html) 是兼顾了MindSpore的效率和灵活的混合模式。用户可通过使用`train_with_func.py`文件来使用该混合模式进行训练。

```shell
python train_with_func.py --model=resnet50 --dataset=cifar10 --dataset_download --epoch_size=10
```

> 注：此为试验性质的训练脚本，仍在改进，在1.8.1或更早版本的MindSpore上使用此模式目前并不稳定。

## 教程

我们提供了系列教程，帮助用户学习如何使用MindCV.

- [了解模型配置](tutorials/learn_about_config_CN.md)
- [模型推理](tutorials/inference_CN.md)
- [自定义数据集上的模型微调训练](tutorials/finetune_CN.md)
- [如何自定义模型]() //coming soon
- [视觉ransformer性能优化]() //coming soon
- [部署推理服务](tutorials/deployment_CN.md)

## 模型列表

目前，MindCV支持以下模型。

<details open>
<summary> 支持模型 </summary>

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

关于模型性能和预训练权重的信息请查看 [configs](./configs) 文件夹。

我们将持续加入更多SoTA模型及其训练策略，敬请关注。

</details>

## 支持算法

<details open>
<summary> 支持算法列表 </summary>

* 数据增强
    * [AutoAugment](https://arxiv.org/abs/1805.09501)
    * [RandAugment](https://arxiv.org/abs/1909.13719)
    * [Repeated Augmentation](https://openaccess.thecvf.com/content_CVPR_2020/papers/Hoffer_Augment_Your_Batch_Improving_Generalization_Through_Instance_Repetition_CVPR_2020_paper.pdf)
    * RandErasing (Cutout)
    * CutMix
    * Mixup
    * RandomResizeCrop
    * Color Jitter, Flip, etc
* 优化器
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
* 学习率调度器
    * Warmup Cosine Decay
    * Step LR
    * Polynomial Decay
    * Exponential Decay
* 正则化
    * Weight Decay
    * Label Smoothing
    * Stochastic Depth (depends on networks)
    * Dropout (depends on networks)
* 损失函数
    * Cross Entropy (w/ class weight and auxiliary  logit support)
    * Binary Cross Entropy  (w/ class weight and auxiliary  logit support)
    * Soft Cross Entropy Loss (automatically enabled if mixup or label smoothing is used)
    * Soft Binary Cross Entropy Loss (automatically enabled if mixup or label smoothing is used)
* 模型融合
    * Warmup EMA (Exponential Moving Average)

</details>

## 日志

### 更新
- 2023/04/28
1. 增添了一些新模型，列出如下：
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
2. 错误修正:
    - 分布式训练时，需对每个进程设置相同的随机数种子
    - 检查YAML配置文件中的选项是否存在于命令行解析器
    - 修正了优化器`Adan`中标志变量不为`Tensor`的错误

- 2023/03/25
1. 更新ResNet网络预训练权重，现在预训练权重有更高Top1精度
    - ResNet18精度从70.09提升到70.31
    - ResNet34精度从73.69提升到74.15
    - ResNet50精度从76.64提升到76.69
    - ResNet101精度从77.63提升到78.24
    - ResNet152精度从78.63提升到78.72
2. 按照规则(model_scale-sha256sum.ckpt)更新预训练权重名字和相应下载URL链接。

- 2023/03/05
1. 增加Lion (EvoLved Sign Momentum)优化器，论文 https://arxiv.org/abs/2302.06675
	- Lion所使用的学习率一般比Adamw小3到10倍，而权重衰减(weigt_decay)要大3到10倍.
2. 增加6个模型及其训练策略、预训练权重：
	- [HRNet](configs/hrnet)
	- [SENet](configs/senet)
	- [GoogLeNet](configs/googlenet)
	- [Inception V3](configs/inception_v3)
	- [Inception V4](configs/inception_v4)
	- [Xception](configs/xception)
3. Support gradient clip

- 2023/01/10
1. MindCV v0.1发布! 支持通过PyPI安装 (`pip install mindcv`).
2. 新增4个模型的预训练权重及其策略： googlenet, inception_v3, inception_v4, xception

- 2022/12/09

1. 支持在所有学习率策略中添加学习率预热操作，除cosine decay策略外。
2. 支持`Repeated Augmenation`操作，可以通过`--aug_repeats`对其进行设置，设置值应大于1(通常为3或4)。
3. 支持EMA。
4. 通过支持mixup和cutmix操作进一步优化BCE损失函数。

- 2022/11/21

1. 支持模型损失和正确率的可视化。
2. 支持伦次维度的cosine decay策略的学习率预热操作（之前仅支持步维度）。

- 2022/11/09

1. 支持2个ViT预训练模型。
2. 支持RandAugment augmentation操作。
3. 提高了CutMix操作的可用性，CutMix和Mixup目前可以一起使用。
4. 解决了学习率画图的bug。

- 2022/10/12

1. BCE和CE损失函数目前都支持class-weight config操作、label smoothing操作、auxilary logit input操作（适用于类似Inception模型）。

- 2022/09/13

1. 支持Adan优化器(试用版)。

### 贡献方式

欢迎开发者用户提issue或提交代码PR，或贡献更多的算法和模型，一起让MindCV变得更好。

有关贡献指南，请参阅[CONTRIBUTING.md](CONTRIBUTING.md)。请遵循[模型编写指南](mindcv/models/model_template_CN.md)所规定的规则来贡献模型接口：)

### 许可证

本项目遵循[Apache License 2.0](License.md)开源协议。

### 致谢

MindCV是由MindSpore团队、西安电子科技大学、西安交通大学联合开发的开源项目。
衷心感谢所有参与的研究人员和开发人员为这个项目所付出的努力。
十分感谢 [OpenI](https://openi.pcl.ac.cn/) 平台所提供的算力资源。

### 引用

如果你觉得MindCV对你的项目有帮助，请考虑引用：

```latex
@misc{MindSpore Computer Vision 2022,
    title={{MindSpore Computer  Vision}:MindSpore Computer Vision Toolbox and Benchmark},
    author={MindSpore Vision Contributors},
    howpublished = {\url{https://github.com/mindspore-lab/mindcv/}},
    year={2022}
}
```
