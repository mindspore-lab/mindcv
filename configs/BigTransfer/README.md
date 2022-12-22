# BigTransfer

> [Big Transfer (BiT): General Visual Representation Learning](https://arxiv.org/abs/1912.11370)

## Introduction

---

Transfer of pre-trained representations improves sample efficiency and simplifies hyperparameter tuning when training deep neural
networks for vision. We revisit the paradigm of pre-training on large supervised datasets and fine-tuning the model on a target task. We scale
up pre-training, and propose a simple recipe that we call Big Transfer
(BiT). By combining a few carefully selected components, and transferring using a simple heuristic, we achieve strong performance on over
20 datasets. BiT performs well across a surprisingly wide range of data
regimes — from 1 example per class to 1 M total examples. BiT achieves
87.5% top-1 accuracy on ILSVRC-2012, 99.4% on CIFAR-10, and 76.3%
on the 19 task Visual Task Adaptation Benchmark (VTAB). On small
datasets, BiT attains 76.8% on ILSVRC-2012 with 10 examples per class,
and 97.0% on CIFAR-10 with 10 examples per class. We conduct detailed
analysis of the main components that lead to high transfer performance.

![BiT](./BiT.png)

## Results

---

|    Model     | Context  | Top-1 (%) | Top-5 (%) | Params(M) |  Train T.  |  Infer T.   |                           Download                           |                            Config                            |                             Log                              |
| :----------: | :------: | :-------: | :-------: | :-------: | :--------: | :---------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|    BiT50-S   | D910x8-G |   76.81   |   93.17   |    25     | 651s/step  |             | [model](https://download.mindspore.cn/toolkits/mindcv/bit/BiTresnet50.ckpt) | [cfg](https://github.com/mindspore-lab/mindcv/blob/main/configs/BigTransfer/BiT50.yaml) | [log](https://github.com/mindspore-lab/mindcv/tree/main/configs/BigTransfer) |

#### Notes

- All models are trained on ImageNet-1K training set and the top-1 accuracy is reported on the validatoin set.
- Context: D910 x 8 - G, D910 - Ascend 910, x8 - 8 devices, G - graph mode.

## Quick Start

---

### Preparation

#### Installation

Please refer to the [installation instruction](https://github.com/mindspore-ecosystem/mindcv#installation) in MindCV.

#### Dataset Preparation

Please download the [ImageNet-1K](https://www.image-net.org/download.php) dataset for model training and validation.

### Training

- **Hyper-parameters.** The hyper-parameter configurations for producing the reported results are stored in the yaml files in `mindcv/configs/BigTransfer` folder. For example, to train with one of these configurations, you can run:

  ```
  # train BiT on 8 GPUs
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # suppose there are 8 GPUs
  mpirun -n 8 python train.py -c configs/BigTransfer/BiT50_ascend.yaml --data_dir /path/to/imagenet
  ```

  Note that the number of GPUs/Ascends and batch size will influence the training results. To reproduce the training result at most, it is recommended to use the **same number of GPUs/Ascends** with the same batch size.

Detailed adjustable parameters and their default value can be seen in [config.py](https://github.com/mindspore-lab/mindcv/tree/main/configs/BigTransfer)

### Validation

- To validate the trained model, you can use `validate.py`. Here is an example for BiT-50 to verify the accuracy of pretrained weights.

  ```
  python validate.py -c configs/BigTransfer/BiT50_ascend.yaml --data_dir /path/to/imagenet --ckpt_path /path/to/ckpt
  ```

### Deployment (optional)

Please refer to the deployment tutorial in MindCV.
