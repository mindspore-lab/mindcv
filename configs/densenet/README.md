# DenseNet
<!--- Guideline: please use url linked to the paper abstract in ArXiv instead of PDF for fast loading.  -->
> [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)

## Introduction
<!--- Guideline: Introduce the model and architectures. Please cite if you use/adopt paper explanation from others. -->
<!--- Guideline: If an architecture table/figure is available in the paper, please put one here and cite for intuitive illustration. -->

Recent work has shown that convolutional networks can be substantially deeper, more accurate, and more efficient to train if
they contain shorter connections between layers close to the input and those close to the output. Dense Convolutional
Network (DenseNet) is introduced based on this observation, which connects each layer to every other layer in a
feed-forward fashion. Whereas traditional convolutional networks with $L$ layers have $L$ connections-one between each
layer and its subsequent layer, DenseNet has $\frac{L(L+1)}{2}$ direct connections. For each layer, the feature maps
of all preceding layers are used as inputs, and their feature maps are used as inputs into all subsequent layers.
DenseNets have several compelling advantages: they alleviate the vanishing-gradient problem, strengthen feature
propagation, encourage feature reuse, and substantially reduce the number of parameters.[[1](#references)]

<p align="center">
  <img src="https://user-images.githubusercontent.com/52945530/210045537-7eda82c7-4575-4820-ba94-8fcab11c6482.png" width=800 />
</p>
<p align="center">
  <em>Figure 1. Architecture of DenseNet [<a href="#references">1</a>] </em>
</p>

## Requirements
| mindspore | ascend driver |  firmware   | cann toolkit/kernel |
| :-------: | :-----------: | :---------: | :-----------------: |
|   2.3.1   |   24.1.RC2    | 7.3.0.1.231 |    8.0.RC2.beta1    |

## Quick Start

### Preparation

#### Installation
Please refer to the [installation instruction](https://mindspore-lab.github.io/mindcv/installation/) in MindCV.

#### Dataset Preparation
Please download the [ImageNet-1K](https://www.image-net.org/challenges/LSVRC/2012/index.php) dataset for model training and validation.

### Training
<!--- Guideline: Please avoid using shell scripts in the command line. Python scripts preferred. -->

* Distributed Training

It is easy to reproduce the reported results with the pre-defined training recipe. For distributed training on multiple Ascend 910 devices, please run

```shell
# distributed training on multiple NPU devices
msrun --bind_core=True --worker_num 8 python train.py --config configs/densenet/densenet_121_ascend.yaml --data_dir /path/to/imagenet
```


For detailed illustration of all hyper-parameters, please refer to [config.py](https://github.com/mindspore-lab/mindcv/blob/main/config.py).

**Note:**  As the global batch size  (batch_size x num_devices) is an important hyper-parameter, it is recommended to keep the global batch size unchanged for reproduction or adjust the learning rate linearly to a new global batch size.

* Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on single NPU device
python train.py --config configs/densenet/densenet_121_ascend.yaml --data_dir /path/to/dataset --distribute False
```

### Validation

To validate the accuracy of the trained model, you can use `validate.py` and parse the checkpoint path with `--ckpt_path`.

```
python validate.py -c configs/densenet/densenet_121_ascend.yaml --data_dir /path/to/imagenet --ckpt_path /path/to/ckpt
```

## Performance

Our reproduced model performance on ImageNet-1K is reported as follows.

Experiments are tested on ascend 910* with mindspore 2.3.1 graph mode.


| model name  | params(M) | cards | batch size | resolution | jit level | graph compile | ms/step | img/s   | acc@top1 | acc@top5 | recipe                                                                                              | weight                                                                                                    |
| ----------- | --------- | ----- | ---------- | ---------- | --------- | ------------- | ------- | ------- | -------- | -------- | --------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| densenet121 | 8.06      | 8     | 32         | 224x224    | O2        | 300s          | 47,34   | 5446.81 | 75.67    | 92.77    | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/densenet/densenet_121_ascend.yaml) | [weights](https://download-mindspore.osinfra.cn/toolkits/mindcv/densenet/densenet121-bf4ab27f-910v2.ckpt) |

Experiments are tested on ascend 910 with mindspore 2.3.1 graph mode.

| model name  | params(M) | cards | batch size | resolution | jit level | graph compile | ms/step | img/s   | acc@top1 | acc@top5 | recipe                                                                                              | weight                                                                                             |
| ----------- | --------- | ----- | ---------- | ---------- | --------- | ------------- | ------- | ------- | -------- | -------- | --------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| densenet121 | 8.06      | 8     | 32         | 224x224    | O2        | 191s          | 43.28   | 5914.97 | 75.64    | 92.84    | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/densenet/densenet_121_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/densenet/densenet121-120_5004_Ascend.ckpt) |

### Notes
- top-1 and top-5: Accuracy reported on the validation set of ImageNet-1K.

## References
<!--- Guideline: Citation format GB/T 7714 is suggested. -->

[1] Huang G, Liu Z, Van Der Maaten L, et al. Densely connected convolutional networks[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2017: 4700-4708.
