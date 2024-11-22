# PiT
> [PiT: Rethinking Spatial Dimensions of Vision Transformers](https://arxiv.org/abs/2103.16302v2)

## Introduction

PiT (Pooling-based Vision Transformer) is an improvement of Vision Transformer (ViT) model proposed by Byeongho Heo in 2021. PiT adds pooling layer on the basis of ViT model, so that the spatial dimension of each layer is reduced like CNN, instead of ViT using the same spatial dimension for all layers. PiT achieves the improved model capability and generalization performance against ViT. [[1](#references)]


<p align="center">
  <img src="https://user-images.githubusercontent.com/37565353/215304821-efaf99ad-12ba-4020-90a3-5897247f9368.png" width=400 />

</p>
<p align="center">
  <em>Figure 1. Architecture of PiT [<a href="#references">1</a>] </em>
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

* Distributed Training

It is easy to reproduce the reported results with the pre-defined training recipe. For distributed training on multiple Ascend 910 devices, please run

```shell
# distributed training on multiple NPU devices
msrun --bind_core=True --worker_num 8 python train.py --config configs/pit/pit_xs_ascend.yaml --data_dir /path/to/imagenet
```




For detailed illustration of all hyper-parameters, please refer to [config.py](https://github.com/mindspore-lab/mindcv/blob/main/config.py).

**Note:**  As the global batch size  (batch_size x num_devices) is an important hyper-parameter, it is recommended to keep the global batch size unchanged for reproduction or adjust the learning rate linearly to a new global batch size.

* Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on single NPU device
python train.py --config configs/pit/pit_xs_ascend.yaml --data_dir /path/to/dataset --distribute False
```

### Validation

To validate the accuracy of the trained model, you can use `validate.py` and parse the checkpoint path with `--ckpt_path`.

```shell
python validate.py -c configs/pit/pit_xs_ascend.yaml --data_dir /path/to/imagenet --ckpt_path /path/to/ckpt
```

## Performance

Our reproduced model performance on ImageNet-1K is reported as follows.

Experiments are tested on ascend 910* with mindspore 2.3.1 graph mode.




| model name | params(M) | cards | batch size | resolution | jit level | graph compile | ms/step | img/s   | acc@top1 | acc@top5 | recipe                                                                                   | weight                                                                                          |
| ---------- | --------- | ----- | ---------- | ---------- | --------- | ------------- | ------- | ------- | -------- | -------- | ---------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| pit_ti     | 4.85      | 8     | 128        | 224x224    | O2        | 212s          | 266.47  | 3842.83 | 73.26    | 91.57    | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/pit/pit_ti_ascend.yaml) | [weights](https://download-mindspore.osinfra.cn/toolkits/mindcv/pit/pit_ti-33466a0d-910v2.ckpt) |




Experiments are tested on ascend 910 with mindspore 2.3.1 graph mode.




| model name | params(M) | cards | batch size | resolution | jit level | graph compile | ms/step | img/s   | acc@top1 | acc@top5 | recipe                                                                                   | weight                                                                            |
| ---------- | --------- | ----- | ---------- | ---------- | --------- | ------------- | ------- | ------- | -------- | -------- | ---------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| pit_ti     | 4.85      | 8     | 128        | 224x224    | O2        | 192s          | 271.50  | 3771.64 | 72.96    | 91.33    | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/pit/pit_ti_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/pit/pit_ti-e647a593.ckpt) |




### Notes
- top-1 and top-5: Accuracy reported on the validation set of ImageNet-1K.


## References

[1] Heo B, Yun S, Han D, et al. Rethinking spatial dimensions of vision transformers[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021: 11936-11945.
