# CoaT

> [Co-Scale Conv-Attentional Image Transformers](https://arxiv.org/abs/2104.06399v2)

##  Introduction

Co-Scale Conv-Attentional Image Transformer (CoaT) is a Transformer-based image classifier equipped with co-scale and conv-attentional mechanisms. First, the co-scale mechanism maintains the integrity of Transformers' encoder branches at individual scales, while allowing representations learned at different scales to effectively communicate with each other. Second, the conv-attentional mechanism is designed by realizing a relative position embedding formulation in the factorized attention module with an efficient convolution-like implementation. CoaT empowers image Transformers with enriched multi-scale and contextual modeling capabilities.

## Results

Our reproduced model performance on ImageNet-1K is reported as follows.

<div align="center">

| Model           | Context   |  Top-1 (%)  |  Top-5 (%)  | Params (M) | Recipe                                                                                   | Weight                                                                            |
|-----------------|-----------|-------|------------|------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| coat_lite_tiny | D910x8-G | 77.35 | 93.43 | 5.72      | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/coat/coat_lite_tiny_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/coat/coat_lite_tiny-fa7bf894.ckpt) |
| coat_lite_mini | D910x8-G | 78.51 | 93.84 | 11.01 | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/coat/coat_lite_mini_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/coat/coat_lite_mini-55a52f05.ckpt) |
| coat_tiny | D910x8-G | 79.67 | 94.88 | 5.50 | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/coat/coat_tiny_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/coat/coat_tiny-071cb792.ckpt) |
| coat_mini | D910x8-G | 81.08 | 95.34 | 10.34 | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/coat/coat_mini_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/coat/coat_mini-57c5bce7.ckpt) |

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
# distributed training on multiple GPU/Ascend devices
mpirun -n 8 python train.py --config configs/coat/coat_lite_tiny_ascend.yaml --data_dir /path/to/imagenet
```

> If the script is executed by the root user, the `--allow-run-as-root` parameter must be added to `mpirun`

Similarly, you can train the model on multiple GPU devices with the above `mpirun` command.

For detailed illustration of all hyper-parameters, please refer to [config.py](https://github.com/mindspore-lab/mindcv/blob/main/config.py).

**Note:** As the global batch size (batch_size x num_devices) is an important hyper-parameter, it is recommended to keep the global batch size unchanged for reproduction or adjust the learning rate linearly to a new global batch size.

- Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on a CPU/GPU/Ascend device
python train.py --config configs/coat/coat_lite_tiny_ascend.yaml --data_dir /path/to/dataset --distribute False
```

### Validation

To validate the accuracy of the trained model, you can use `validate.py` and parse the checkpoint path with `--ckpt_path`.

```shell
python validate.py -c configs/coat/coat_lite_tiny_ascend.yaml --data_dir /path/to/imagenet --ckpt_path /path/to/ckpt
```

### Deployment

To deploy online inference services with the trained model efficiently, please refer to the [deployment tutorial](https://mindspore-lab.github.io/mindcv/tutorials/deployment/).

## References

[1] Han D, Yun S, Heo B, et al. Rethinking channel dimensions for efficient model design[C]//Proceedings of the IEEE/CVF conference on Computer Vision and Pattern Recognition. 2021: 732-741.
