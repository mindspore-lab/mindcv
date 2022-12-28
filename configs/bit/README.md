# BigTransfer

> [Big Transfer (BiT): General Visual Representation Learning](https://arxiv.org/abs/1912.11370)

## Introduction


Transfer of pre-trained representations improves sample efficiency and simplifies hyperparameter tuning when training deep neural
networks for vision. This paper scales up pre-training and proposes a simple recipe that calls Big Transfer
(BiT). By combining a few carefully selected components, and transferring using a simple heuristic, they achieve strong performance on over
20 datasets. BiT performs well across a surprisingly wide range of data
regimes — from 1 example per class to 1 M total examples. BiT achieves
87.5% top-1 accuracy on ILSVRC-2012, 99.4% on CIFAR-10, and 76.3%
on the 19 task Visual Task Adaptation Benchmark (VTAB). On small
datasets, BiT attains 76.8% on ILSVRC-2012 with 10 examples per class,
and 97.0% on CIFAR-10 with 10 examples per class. 

<p align="center">
  <img src="https://github.com/mindspore-lab/mindcv/blob/main/configs/bit/bit.png" width=800 />  
</p>
<p align="center">
  <em>Figure 1. Architecture of BiT</em>
</p>

## Results

<div align="center">

|       Model      | Context  | Top-1 (%) | Top-5 (%) | Params(M) |                                                 Recipe                                                  |                                    Download                                   |                 
| ---------------- | -------- | --------- | --------- | --------- | ------------------------------------------------------------------------------------------------------- |  ---------------------------------------------------------------------------- |
|  bit_resnet50-S  | D910x8-G |   76.81   |   93.17   |   25.55   |  [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/bit/bit_resnet50_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/bit/BiTresnet50.ckpt) | 

</div>

#### Notes

- Context: Training context denoted as {device}x{pieces}-{MS mode}, where mindspore mode can be G - graph mode or F - pynative mode with ms function. For example, D910x8-G is for training on 8 pieces of Ascend 910 NPU using graph mode. 
- Top-1 and Top-5: Accuracy reported on the validation set of ImageNet-1K. 

## Quick Start

### Preparation

#### Installation

Please refer to the [installation instruction](https://github.com/mindspore-lab/mindcv#installation) in MindCV.

#### Dataset Preparation

Please download the [ImageNet-1K](https://www.image-net.org/challenges/LSVRC/2012/index.php) dataset for model training and validation.

### Training

* Distributed Training

It is easy to reproduce the reported results with the pre-defined training recipe. For distributed training on multiple Ascend 910 devices, please run

```shell
# distrubted training on multiple GPU/Ascend devices
mpirun -n 8 python train.py --config configs/bit/bit_resnet50_ascend.yaml --data_dir /path/to/imagenet
```
  
Similarly, you can train the model on multiple GPU devices with the above `mpirun` command.

For detailed illustration of all hyper-parameters, please refer to [config.py](https://github.com/mindspore-lab/mindcv/blob/main/config.py).

**Note:**  As the global batch size  (batch_size x num_devices) is an important hyper-parameter, it is recommended to keep the global batch size unchanged for reproduction or adjust the learning rate linearly to a new global batch size.

* Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on a CPU/GPU/Ascend device
python train.py --config configs/bit/bit_resnet50_ascend.yaml --data_dir /path/to/dataset --distribute False
```

### Validation

To validate the accuracy of the trained model, you can use `validate.py` and parse the checkpoint path with `--ckpt_path`.

```
python validate.py -c configs/bit/bit_resnet50_ascend.yaml --data_dir /path/to/imagenet --ckpt_path /path/to/ckpt
```

### Deployment

Please refer to the [deployment tutorial](https://github.com/mindspore-lab/mindcv/blob/main/tutorials/deployment.md) in MindCV.
