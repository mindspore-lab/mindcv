# SqueezeNet

***
> [SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size](https://arxiv.org/pdf/1602.07360.pdf)

## Introduction

***
SqueezeNet is a smaller CNN architectures which is comprised mainly of Fire modules and it achieves AlexNet-level
accuracy on ImageNet with 50x fewer parameters. SqueezeNet can offer at least three advantages: (1) Smaller CNNs require
less communication across servers during distributed training. (2) Smaller CNNs require less bandwidth to export a new
model from the cloud to an autonomous car. (3) Smaller CNNs are more feasible to deploy on FPGAs and other hardware with
limited memory. Additionally, with model compression techniques, SqueezeNet is able to be compressed to less than
0.5MB (510× smaller than AlexNet). Blow is macroarchitectural view of SqueezeNet architecture. Left: SqueezeNet ;
Middle: SqueezeNet with simple bypass; Right: SqueezeNet with complex bypass .

![](squeezenet.png)

## Results
***
|Model|Context| Top1/Top5 | Params(M) |Ckpt|Config|
| :------:| :------: | :-------: | :-------: |:-----: |:-----: |
|squeezenet_1.0| GPUx8-G |59.49/81.22| |[ckpt](https://download.mindspore.cn/toolkits/mindcv/squeezenet/squeezenet_1.0_224.ckpt)|[yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/squeezenet/squeezenet_1.0_gpu.yaml)|
squeezenet_1.1 | D910x8-G |58.99/80.99 |  | [ckpt](https://download.mindspore.cn/toolkits/mindcv/squeezenet/squeezenet_1.1_224.ckpt) | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/squeezenet/squeezenet_1.1_gpu.yaml) | 


#### Notes
- Context: D910 -> HUAWEI Ascend 910 |  x 8 ->  using 8 NPUs | G -> MindSpore graph model ; F -> MindSpore pynative mode.

## Quick Start

---

### Preparation

#### Installation

Please refer to the [installation instruction](https://github.com/mindspore-lab/mindcv#installation) in MindCV.

#### Dataset Preparation

Please download the [ImageNet-1K](https://www.image-net.org/download.php) dataset for model training and validation.

### Training

- **Hyper-parameters.** The hyper-parameter configurations for producing the reported results are stored in the yaml files in `mindcv/configs/vit` folder. For example, to train with one of these configurations, you can run:

  ```
  # train squeezenet on 8 GPU
  mpirun -n 8 python train.py -c configs/squeezenet/squeezenet_1.0_gpu.yaml --data_dir /path/to/imagenet
  ```

  Note that the number of GPUs/Ascends and batch size will influence the training results. To reproduce the training result at most, it is recommended to use the **same number of GPUs/NPUs** with the same batch size.

Detailed adjustable parameters and their default value can be seen in [config.py](https://github.com/mindspore-lab/mindcv/blob/main/config.py)

### Validation

- To validate the trained model, you can use `validate.py`. Here is an example for vit_b_32 to verify the accuracy of pretrained weights.

  ```
  python validate.py -c configs/squeezenet/squeezenet_1.0_gpu.yaml --data_dir /path/to/imagenet --ckpt_path /path/to/ckpt
