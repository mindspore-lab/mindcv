# MnasNet
> [MnasNet: Platform-Aware Neural Architecture Search for Mobile](https://arxiv.org/abs/1807.11626)

## Introduction

Designing convolutional neural networks (CNN) for mobile devices is challenging because mobile models need to be small and fast, yet still accurate. Although significant efforts have been dedicated to design and improve mobile CNNs on all dimensions, it is very difficult to manually balance these trade-offs when there are so many architectural possibilities to consider. In this paper, the authors propose an automated mobile neural architecture search (MNAS) approach, which explicitly incorporate model latency into the main objective so that the search can identify a model that achieves a good trade-off between accuracy and latency. Unlike previous work, where latency is considered via another, often inaccurate proxy (e.g., FLOPS), our approach directly measures real-world inference latency by executing the model on mobile phones. To further strike the right balance between flexibility and search space size, the authors propose a novel factorized hierarchical search space that encourages layer diversity throughout the network.[[1](#references)]

<p align="center">
  <img src="https://user-images.githubusercontent.com/53842165/210044057-35febc60-8d24-434a-a4f2-db8db3859e7a.png" width=800 />
</p>
<p align="center">
  <em>Figure 1. Architecture of MnasNet [<a href="#references">1</a>] </em>
</p>

## Results

Our reproduced model performance on ImageNet-1K is reported as follows.

<div align="center">

| Model       | Context  | Top-1 (%) | Top-5 (%) | Params (M) | Recipe                                                                                            | Download                                                                                 |
|-------------|----------|-----------|-----------|------------|---------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------|
| mnasnet_050 | D910x8-G | 68.07     | 88.09     | 2.14       | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/mnasnet/mnasnet_0.5_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/mnasnet/mnasnet_050-7d8bf4db.ckpt) |
| mnasnet_075 | D910x8-G | 71.81     | 90.53     | 3.20       | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/mnasnet/mnasnet_0.75_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/mnasnet/mnasnet_075-465d366d.ckpt) |
| mnasnet_100 | D910x8-G | 74.28     | 91.70     | 4.42       | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/mnasnet/mnasnet_1.0_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/mnasnet/mnasnet_100-1bcf43f8.ckpt)  |
| mnasnet_130 | D910x8-G | 75.65     | 92.64     | 6.33       | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/mnasnet/mnasnet_1.3_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/mnasnet/mnasnet_130-a43a150a.ckpt)  |
| mnasnet_140 | D910x8-G | 76.01     | 92.83     | 7.16       | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/mnasnet/mnasnet_1.4_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/mnasnet/mnasnet_140-7e20bb30.ckpt)  |

</div>

#### Notes

- Context: Training context denoted as {device}x{pieces}-{MS mode}, where mindspore mode can be G - graph mode or F - pynative mode with ms function. For example, D910x8-G is for training on 8 pieces of Ascend 910 NPU using graph mode.
- Top-1 and Top-5: Accuracy reported on the validation set of ImageNet-1K.

## Quick Start

### Preparation

#### Installation
Please refer to the [installation instruction](https://github.com/mindspore-ecosystem/mindcv#installation) in MindCV.

#### Dataset Preparation
Please download the [ImageNet-1K](https://www.image-net.org/challenges/LSVRC/2012/index.php) dataset for model training and validation.

### Training

* Distributed Training

It is easy to reproduce the reported results with the pre-defined training recipe. For distributed training on multiple Ascend 910 devices, please run

```shell
# distributed training on multiple GPU/Ascend devices
mpirun -n 8 python train.py --config configs/mnasnet/mnasnet_0.75_ascend.yaml --data_dir /path/to/imagenet
```

> If the script is executed by the root user, the `--allow-run-as-root` parameter must be added to `mpirun`.

Similarly, you can train the model on multiple GPU devices with the above `mpirun` command.

For detailed illustration of all hyper-parameters, please refer to [config.py](https://github.com/mindspore-lab/mindcv/blob/main/config.py).

**Note:**  As the global batch size  (batch_size x num_devices) is an important hyper-parameter, it is recommended to keep the global batch size unchanged for reproduction or adjust the learning rate linearly to a new global batch size.

* Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on a CPU/GPU/Ascend device
python train.py --config configs/mnasnet/mnasnet_0.75_ascend.yaml --data_dir /path/to/dataset --distribute False
```

### Validation

To validate the accuracy of the trained model, you can use `validate.py` and parse the checkpoint path with `--ckpt_path`.

```shell
python validate.py -c configs/mnasnet/mnasnet_0.75_ascend.yaml --data_dir /path/to/imagenet --ckpt_path /path/to/ckpt
```

### Deployment

Please refer to the [deployment tutorial](https://mindspore-lab.github.io/mindcv/tutorials/deployment/) in MindCV.

## References

[1] Tan M, Chen B, Pang R, et al. Mnasnet: Platform-aware neural architecture search for mobile[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019: 2820-2828.
