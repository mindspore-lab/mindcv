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

## Requirements
| mindspore | ascend driver |  firmware   | cann toolkit/kernel |
| :-------: | :-----------: | :---------: | :-----------------: |
|   2.5.0   |   24.1.0      | 7.5.0.3.220 |     8.0.0.beta1     |



## Quick Start

### Preparation

#### Installation
Please refer to the [installation instruction](https://mindspore-lab.github.io/mindcv/installation/) in MindCV.

#### Dataset Preparation
Please download the [ImageNet-1K](https://www.image-net.org/challenges/LSVRC/2012/index.php) dataset for model training and validation.

### Training

* Distributed Training

It is easy to reproduce the reported results with the pre-defined training recipe. For distributed training on multiple Ascend 910 devices, please run

```shell
# distributed training on multiple NPU devices
msrun --bind_core=True --worker_num 8 python train.py --config configs/mnasnet/mnasnet_0.75_ascend.yaml --data_dir /path/to/imagenet
```


For detailed illustration of all hyper-parameters, please refer to [config.py](https://github.com/mindspore-lab/mindcv/blob/main/config.py).

**Note:**  As the global batch size  (batch_size x num_devices) is an important hyper-parameter, it is recommended to keep the global batch size unchanged for reproduction or adjust the learning rate linearly to a new global batch size.

* Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on single NPU device
python train.py --config configs/mnasnet/mnasnet_0.75_ascend.yaml --data_dir /path/to/dataset --distribute False
```

### Validation

To validate the accuracy of the trained model, you can use `validate.py` and parse the checkpoint path with `--ckpt_path`.

```shell
python validate.py -c configs/mnasnet/mnasnet_0.75_ascend.yaml --data_dir /path/to/imagenet --ckpt_path /path/to/ckpt
```

## Performance

Our reproduced model performance on ImageNet-1K is reported as follows.

Experiments are tested on ascend 910* with mindspore 2.5.0 graph mode.


| model name  | params(M) | cards | batch size | resolution | jit level | graph compile | ms/step | img/s    | acc@top1 | acc@top5 | recipe                                                                                             | weight                                                                                                   |
| ----------- | --------- | ----- | ---------- | ---------- | --------- | ------------- |---------| -------- | -------- | -------- | -------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| mnasnet_075 | 3.20      | 8     | 256        | 224x224    | O2        | 144s          | 175.6   | 11662.87 | 71.77    | 90.52    | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/mnasnet/mnasnet_0.75_ascend.yaml) | [weights](https://download-mindspore.osinfra.cn/toolkits/mindcv/mnasnet/mnasnet_075-083b2bc4-910v2.ckpt) |


### Notes
- top-1 and top-5: Accuracy reported on the validation set of ImageNet-1K.

## References

[1] Tan M, Chen B, Pang R, et al. Mnasnet: Platform-aware neural architecture search for mobile[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019: 2820-2828.
