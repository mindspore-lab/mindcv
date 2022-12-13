# ShuffleNetV2
> [ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://arxiv.org/pdf/1807.11164.pdf)

## Introduction
***

A key point was raised in ShuffleNetV2, where previous lightweight networks were guided by computing an indirect measure of network complexity, namely FLOPs. The speed of lightweight networks is described by calculating the amount of floating point operations. But the speed of operation was never considered directly. The running speed in mobile devices needs to consider not only FLOPs, but also other factors such as memory accesscost and platform characterics.

Therefore, based on these two principles, ShuffleNetV2 proposes four effective network design principles.

- MAC is minimized when the input feature matrix of the convolutional layer is equal to the output feature matrixchannel (when FLOPs are kept constant).
- MAC increases when the groups of GConv increase (while keeping FLOPs constant).
- the higher the fragmentation of the network design, the slower the speed.
- The impact of Element-wise operation is not negligible.

![](./ShuffleNetV2_Block.png)

## Results
***

| Model              | Context  | Top-1 (%) | Top-5 (%) | Params (M) | Train T.   | Infer T. | Download  | Config  | Log     |
| ------------------ | -------- | --------- | --------- | ---------- | ---------- | -------- | --------- | ------- | ------- |
| shufflenet_v2_x0_5 | D910x8-G | 60.68     | 82.44     | 1.37       | 99s/epoch  |          | [model]() | [cfg]() | [log]() |
| shufflenet_v2_x1_0 | D910x8-G | 69.51     | 88.67     | 2.29       | 101s/epoch |          | [model]() | [cfg]() | [log]() |
| shufflenet_v2_x1_5 | D910x8-G | 72.59     | 90.79     | 3.53       | 125s/epoch |          | [model]() | [cfg]() | [log]() |
| shufflenet_v2_x2_0 | D910x8-G | 75.14     | 92.13     | 7.44       | 149s/epoch |          | [model]() | [cfg]() | [log]() |

#### Notes

- All models are trained on ImageNet-1K training set and the top-1 accuracy is reported on the validatoin set.
- Context: GPU_TYPE x pieces - G/F, G - graph mode, F - pynative mode with ms function.  

## Quick Start
***
### Preparation

#### Installation
Please refer to the [installation instruction](https://github.com/mindspore-ecosystem/mindcv#installation) in MindCV.

#### Dataset Preparation
Please download the [ImageNet-1K](https://www.image-net.org/download.php) dataset for model training and validation.

### Training

- **Hyper-parameters.** The hyper-parameter configurations for producing the reported results are stored in the yaml files in `mindcv/configs/shufflenetv2` folder. For example, to train with one of these configurations, you can run:

  ```shell
  # train shufflenet_v2_x2_0 on Ascend
  cd mindcv/scripts
  bash run_distribute_train_ascend.sh ./hccl_8p_01234567_123.60.231.9.json /tmp/dataset/imagenet ../configs/shufflenet_v2/shufflenet_v2_x2.0_ascend.yaml
  ```

  Note that the number of GPUs/Ascends and batch size will influence the training results. To reproduce the training result at most, it is recommended to use the **same number of GPUs/Ascneds** with the same batch size.

Detailed adjustable parameters and their default value can be seen in [config.py](../../config.py).

### Validation

- To validate the trained model, you can use `validate.py`. Here is an example for shufflenet_v2_x2_0 to verify the accuracy of pretrained weights.

  ```shell
  python validate.py -c configs/shufflenet_v2/shufflenet_v2_x2.0_ascend.yaml
  ```

### Deployment (optional)

Please refer to the deployment tutorial in MindCV.

