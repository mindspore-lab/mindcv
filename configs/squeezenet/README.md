# SqueezeNet

> [SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size](https://arxiv.org/abs/1602.07360)



## Introduction

SqueezeNet is a smaller CNN architectures which is comprised mainly of Fire modules and it achieves AlexNet-level
accuracy on ImageNet with 50x fewer parameters. SqueezeNet can offer at least three advantages: (1) Smaller CNNs require
less communication across servers during distributed training. (2) Smaller CNNs require less bandwidth to export a new
model from the cloud to an autonomous car. (3) Smaller CNNs are more feasible to deploy on FPGAs and other hardware with
limited memory. Additionally, with model compression techniques, SqueezeNet is able to be compressed to less than
0.5MB (510× smaller than AlexNet). Blow is macroarchitectural view of SqueezeNet architecture. Left: SqueezeNet ;
Middle: SqueezeNet with simple bypass; Right: SqueezeNet with complex bypass.

<p align="center">
  <img src="https://user-images.githubusercontent.com/8156835/210043440-5218ae6d-54ed-4d1e-a4ff-335d4a2baaa5.png" width=800 />
</p>
<p align="center">
  <em>Figure 1. Architecture of SqueezeNet [<a href="#references">1</a>] </em>
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

Please download the [ImageNet-1K](https://www.image-net.org/challenges/LSVRC/2012/index.php) dataset for model training
and validation.

### Training

<!--- Guideline: Avoid using shell script in the command line. Python script preferred. -->

* Distributed Training

It is easy to reproduce the reported results with the pre-defined training recipe. For distributed training on multiple
Ascend 910 devices, please run

```shell
# distributed training on multiple NPU devices
msrun --bind_core=True --worker_num 8 python train.py --config configs/squeezenet/squeezenet_1.0_ascend.yaml --data_dir /path/to/imagenet
```


For detailed illustration of all hyper-parameters, please refer
to [config.py](https://github.com/mindspore-lab/mindcv/blob/main/config.py).

**Note:**  As the global batch size  (batch_size x num_devices) is an important hyper-parameter, it is recommended to
keep the global batch size unchanged for reproduction or adjust the learning rate linearly to a new global batch size.

* Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on single NPU device
python train.py --config configs/squeezenet/squeezenet_1.0_ascend.yaml --data_dir /path/to/dataset --distribute False
```

### Validation

To validate the accuracy of the trained model, you can use `validate.py` and parse the checkpoint path
with `--ckpt_path`.

```
python validate.py -c configs/squeezenet/squeezenet_1.0_ascend.yaml --data_dir /path/to/imagenet --ckpt_path /path/to/ckpt
```

## Performance

Our reproduced model performance on ImageNet-1K is reported as follows.

Experiments are tested on ascend 910* with mindspore 2.3.1 graph mode.




| model name    | params(M) | cards | batch size | resolution | jit level | graph compile | ms/step | img/s    | acc@top1 | acc@top5 | recipe                                                                                                  | weight                                                                                                        |
| ------------- | --------- | ----- | ---------- | ---------- | --------- |---------------| ------- | -------- | -------- | -------- | ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| squeezenet1_0 | 1.25      | 8     | 32         | 224x224    | O2        | 64s           | 23.48   | 10902.90 | 58.75    | 80.76    | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/squeezenet/squeezenet_1.0_ascend.yaml) | [weights](https://download-mindspore.osinfra.cn/toolkits/mindcv/squeezenet/squeezenet1_0-24010b28-910v2.ckpt) |



Experiments are tested on ascend 910 with mindspore 2.3.1 graph mode.




| model name    | params(M) | cards | batch size | resolution | jit level | graph compile | ms/step | img/s    | acc@top1 | acc@top5 | recipe                                                                                                  | weight                                                                                          |
| ------------- | --------- | ----- | ---------- | ---------- | --------- | ------------- | ------- | -------- | -------- | -------- | ------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| squeezenet1_0 | 1.25      | 8     | 32         | 224x224    | O2        | 45s           | 22.36   | 11449.02 | 58.67    | 80.61    | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/squeezenet/squeezenet_1.0_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/squeezenet/squeezenet1_0-eb911778.ckpt) |



### Notes

- top-1 and top-5: Accuracy reported on the validation set of ImageNet-1K.

## References

<!--- Guideline: Citation format GB/T 7714 is suggested. -->
[1] Iandola F N, Han S, Moskewicz M W, et al. SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and< 0.5 MB
model size[J]. arXiv preprint arXiv:1602.07360, 2016.
