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

## Results
<!--- Guideline:
Table Format:
- Model: model name in lower case with _ seperator.
- Context: Training context denoted as {device}x{pieces}-{MS mode}, where mindspore mode can be G - graph mode or F - pynative mode with ms function. For example, D910x8-G is for training on 8 pieces of Ascend 910 NPU using graph mode.
- Top-1 and Top-5: Keep 2 digits after the decimal point.
- Params (M): # of model parameters in millions (10^6). Keep 2 digits after the decimal point
- Recipe: Training recipe/configuration linked to a yaml config file. Use absolute url path.
- Download: url of the pretrained model weights. Use absolute url path.
-->

Our reproduced model performance on ImageNet-1K is reported as follows.

<div align="center">

| Model       | Context  | Top-1 (%) | Top-5 (%) | Params (M) | Recipe                                                                                              | Download                                                                                           |
|-------------|----------|-----------|-----------|------------|-----------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|
| densenet121 | D910x8-G | 75.64     | 92.84     | 8.06       | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/densenet/densenet_121_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/densenet/densenet121-120_5004_Ascend.ckpt) |
| densenet161 | D910x8-G | 79.09     | 94.66     | 28.90      | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/densenet/densenet_161_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/densenet/densenet161-120_5004_Ascend.ckpt) |
| densenet169 | D910x8-G | 77.26     | 93.71     | 14.31      | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/densenet/densenet_169_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/densenet/densenet169-120_5004_Ascend.ckpt) |
| densenet201 | D910x8-G | 78.14     | 94.08     | 20.24      | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/densenet/densenet_201_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/densenet/densenet201-120_5004_Ascend.ckpt) |

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
<!--- Guideline: Please avoid using shell scripts in the command line. Python scripts preferred. -->

* Distributed Training

It is easy to reproduce the reported results with the pre-defined training recipe. For distributed training on multiple Ascend 910 devices, please run

```shell
# distributed training on multiple GPU/Ascend devices
mpirun -n 8 python train.py --config configs/densenet/densenet_121_ascend.yaml --data_dir /path/to/imagenet
```
> If the script is executed by the root user, the `--allow-run-as-root` parameter must be added to `mpirun`.

Similarly, you can train the model on multiple GPU devices with the above `mpirun` command.

For detailed illustration of all hyper-parameters, please refer to [config.py](https://github.com/mindspore-lab/mindcv/blob/main/config.py).

**Note:**  As the global batch size  (batch_size x num_devices) is an important hyper-parameter, it is recommended to keep the global batch size unchanged for reproduction or adjust the learning rate linearly to a new global batch size.

* Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on a CPU/GPU/Ascend device
python train.py --config configs/densenet/densenet_121_ascend.yaml --data_dir /path/to/dataset --distribute False
```

### Validation

To validate the accuracy of the trained model, you can use `validate.py` and parse the checkpoint path with `--ckpt_path`.

```
python validate.py -c configs/densenet/densenet_121_ascend.yaml --data_dir /path/to/imagenet --ckpt_path /path/to/ckpt
```

### Deployment

Please refer to the [deployment tutorial](https://mindspore-lab.github.io/mindcv/tutorials/deployment/) in MindCV.

## References
<!--- Guideline: Citation format GB/T 7714 is suggested. -->

[1] Huang G, Liu Z, Van Der Maaten L, et al. Densely connected convolutional networks[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2017: 4700-4708.
