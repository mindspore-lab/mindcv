# ShuffleNetV2
> [ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://arxiv.org/abs/1807.11164)

## Introduction

A key point was raised in ShuffleNetV2, where previous lightweight networks were guided by computing an indirect measure of network complexity, namely FLOPs. The speed of lightweight networks is described by calculating the amount of floating point operations. But the speed of operation was never considered directly. The running speed in mobile devices needs to consider not only FLOPs, but also other factors such as memory accesscost and platform characterics.

Therefore, based on these two principles, ShuffleNetV2 proposes four effective network design principles.

- MAC is minimized when the input feature matrix of the convolutional layer is equal to the output feature matrixchannel (when FLOPs are kept constant).
- MAC increases when the groups of GConv increase (while keeping FLOPs constant).
- the higher the fragmentation of the network design, the slower the speed.
- The impact of Element-wise operation is not negligible.

<p align="center">
  <img src="https://user-images.githubusercontent.com/8156835/210043336-305a167f-d669-42e7-8b94-eef8e7b78a48.png" width=800 />
</p>
<p align="center">
  <em>Figure 1. Architecture Design in ShuffleNetV2 [<a href="#references">1</a>] </em>
</p>


## Results

Our reproduced model performance on ImageNet-1K is reported as follows.

<div align="center">

| Model            | Context  | Top-1 (%) | Top-5 (%) | Params (M) | Recipe                                                                                                       | Download                                                                                                          |
|------------------|----------|-----------|-----------|------------|--------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| shufflenet_v2_x0_5 | D910x8-G | 60.53     | 82.11     | 1.37       | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/shufflenetv2/shufflenet_v2_0.5_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/shufflenet/shufflenetv2/shufflenet_v2_x0_5-8c841061.ckpt) |
| shufflenet_v2_x1_0 | D910x8-G | 69.47     | 88.88     | 2.29       | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/shufflenetv2/shufflenet_v2_1.0_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/shufflenet/shufflenetv2/shufflenet_v2_x1_0-0da4b7fa.ckpt) |
| shufflenet_v2_x1_5 | D910x8-G | 72.79     | 90.93     | 3.53       | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/shufflenetv2/shufflenet_v2_1.5_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/shufflenet/shufflenetv2/shufflenet_v2_x1_5-00b56131.ckpt) |
| shufflenet_v2_x2_0 | D910x8-G | 75.07     | 92.08     | 7.44       | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/shufflenetv2/shufflenet_v2_2.0_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/shufflenet/shufflenetv2/shufflenet_v2_x2_0-ed8e698d.ckpt) |

</div>

#### Notes
- Context: Training context denoted as {device}x{pieces}-{MS mode}, where mindspore mode can be G - graph mode or F - pynative mode with ms function. For example, D910x8-G is for training on 8 pieces of Ascend 910 NPU using graph mode.
- Top-1 and Top-5: Accuracy reported on the validation set of ImageNet-1K.

#### Notes

- All models are trained on ImageNet-1K training set and the top-1 accuracy is reported on the validatoin set.
- Context: GPU_TYPE x pieces - G/F, G - graph mode, F - pynative mode with ms function.


## Quick Start
### Preparation

#### Installation
Please refer to the [installation instruction](https://github.com/mindspore-lab/mindcv#installation) in MindCV.

#### Dataset Preparation
Please download the [ImageNet-1K](https://www.image-net.org/challenges/LSVRC/2012/index.php) dataset for model training and validation.

### Training
<!--- Guideline: Avoid using shell script in the command line. Python script preferred. -->

* Distributed Training

It is easy to reproduce the reported results with the pre-defined training recipe. For distributed training on multiple Ascend 910 devices, please run

```shell
# distributed training on multiple GPU/Ascend devices
mpirun -n 8 python train.py --config configs/shufflenetv2/shufflenet_v2_0.5_ascend.yaml --data_dir /path/to/imagenet
```

Similarly, you can train the model on multiple GPU devices with the above `mpirun` command.

For detailed illustration of all hyper-parameters, please refer to [config.py](https://github.com/mindspore-lab/mindcv/blob/main/config.py).

**Note:**  As the global batch size  (batch_size x num_devices) is an important hyper-parameter, it is recommended to keep the global batch size unchanged for reproduction or adjust the learning rate linearly to a new global batch size.

* Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on a CPU/GPU/Ascend device
python train.py --config configs/shufflenetv2/shufflenet_v2_0.5_ascend.yaml --data_dir /path/to/dataset --distribute False
```

### Validation

To validate the accuracy of the trained model, you can use `validate.py` and parse the checkpoint path with `--ckpt_path`.

```
python validate.py -c configs/shufflenetv2/shufflenet_v2_0.5_ascend.yaml --data_dir /path/to/imagenet --ckpt_path /path/to/ckpt
```

### Deployment

To deploy online inference services with the trained model efficiently, please refer to the [deployment tutorial](https://mindspore-lab.github.io/mindcv/tutorials/deployment/).

## References
<!--- Guideline: Citation format GB/T 7714 is suggested. -->
[1] Ma N, Zhang X, Zheng H T, et al. Shufflenet v2: Practical guidelines for efficient cnn architecture design[C]//Proceedings of the European conference on computer vision (ECCV). 2018: 116-131.
