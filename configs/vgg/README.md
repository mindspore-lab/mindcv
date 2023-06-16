# VGGNet
<!--- Guideline: please use url linked to the paper abstract in ArXiv instead of PDF for fast loading.  -->
> [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

## Introduction
<!--- Guideline: Introduce the model and architectures. Please cite if you use/adopt paper explanation from others. -->
<!--- Guideline: If an architecture table/figure is available in the paper, please put one here and cite for intuitive illustration. -->

Figure 1 shows the model architecture of VGGNet. VGGNet is a key milestone on image classification task. It expands the model to 16-19 layers for the first time. The key motivation of this model is
that it shows usage of 3x3 kernels is efficient and by adding 3x3 kernels, it could have the same effect as 5x5 or 7x7 kernels. VGGNet could achieve better model performance compared with previous
methods such as GoogleLeNet and AlexNet on ImageNet-1K dataset.[[1](#references)]

<p align="center">
  <img src="https://user-images.githubusercontent.com/77485245/223675336-ca8b0411-86fd-4134-9b37-e601ff82f64b.jpeg" width=800 />
</p>
<p align="center">
  <em>Figure 1. Architecture of VGG [<a href="#references">1</a>] </em>
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

| Model | Context  | Top-1 (%) | Top-5 (%) | Params (M) | Recipe                                                                                   | Download                                                                         |
|-------|----------|-----------|-----------|------------|------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| vgg11 | D910x8-G | 71.86     | 90.50     | 132.86      | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/vgg/vgg11_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/vgg/vgg11-ef31d161.ckpt) |
| vgg13 | D910x8-G | 72.87     | 91.02     | 133.04      | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/vgg/vgg13_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/vgg/vgg13-da805e6e.ckpt) |
| vgg16 | D910x8-G | 74.61     | 91.87     | 138.35      | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/vgg/vgg16_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/vgg/vgg16-95697531.ckpt) |
| vgg19 | D910x8-G | 75.21     | 92.56     | 143.66      | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/vgg/vgg19_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/vgg/vgg19-bedee7b6.ckpt) |

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
# distrubted training on multiple GPU/Ascend devices
mpirun -n 8 python train.py --config configs/vgg/vgg16_ascend.yaml --data_dir /path/to/imagenet
```
> If the script is executed by the root user, the `--allow-run-as-root` parameter must be added to `mpirun`.

Similarly, you can train the model on multiple GPU devices with the above `mpirun` command.

For detailed illustration of all hyper-parameters, please refer to [config.py](https://github.com/mindspore-lab/mindcv/blob/main/config.py).

**Note:**  As the global batch size  (batch_size x num_devices) is an important hyper-parameter, it is recommended to keep the global batch size unchanged for reproduction or adjust the learning rate linearly to a new global batch size.

* Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on a CPU/GPU/Ascend device
python train.py --config configs/vgg/vgg16_ascend.yaml --data_dir /path/to/dataset --distribute False
```

### Validation

To validate the accuracy of the trained model, you can use `validate.py` and parse the checkpoint path with `--ckpt_path`.

```
python validate.py -c configs/vgg/vgg16_ascend.yaml --data_dir /path/to/imagenet --ckpt_path /path/to/ckpt
```

### Deployment

Please refer to the [deployment tutorial](https://mindspore-lab.github.io/mindcv/tutorials/deployment/) in MindCV.

## References
<!--- Guideline: Citation format GB/T 7714 is suggested. -->

[1] Simonyan K, Zisserman A. Very deep convolutional networks for large-scale image recognition[J]. arXiv preprint arXiv:1409.1556, 2014.
