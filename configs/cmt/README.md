# CMT: Convolutional Neural Networks Meet Vision Transformers

> [CMT: Convolutional Neural Networks Meet Vision Transformers](https://arxiv.org/abs/2107.06263)

## Introduction

CMT is a method to make full use of the advantages of CNN and transformers so that the model could capture long-range
dependencies and extract local information. In addition, to reduce computation cost, this method use lightweight MHSA(multi-head self-attention)
and depthwise convolution and pointwise convolution like MobileNet. By combing these parts, CMT could get a SOTA performance
on ImageNet-1K dataset.


## Results

Our reproduced model performance on ImageNet-1K is reported as follows.

<div align="center">

| Model     | Context  | Top-1 (%) | Top-5 (%) | Params(M) | Recipe                                                                                      | Download                                                                             |
|-----------| -------- |-----------|-----------|-----------|---------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------|
| cmt_small | D910x8-G | 83.24     | 96.41     | 26.09     | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/cmt/cmt_small_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/cmt/cmt_small-6858ee22.ckpt) |


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
mpirun -n 8 python train.py --config configs/cmt/cmt_small_ascend.yaml --data_dir /path/to/imagenet
```
> If the script is executed by the root user, the `--allow-run-as-root` parameter must be added to `mpirun`.

Similarly, you can train the model on multiple GPU devices with the above `mpirun` command.

For detailed illustration of all hyper-parameters, please refer to [config.py](https://github.com/mindspore-lab/mindcv/blob/main/config.py).

**Note:**  As the global batch size  (batch_size x num_devices) is an important hyper-parameter, it is recommended to keep the global batch size unchanged for reproduction or adjust the learning rate linearly to a new global batch size.

* Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on a CPU/GPU/Ascend device
python train.py --config configs/cmt/cmt_small_ascend.yaml --data_dir /path/to/dataset --distribute False
```

### Validation

To validate the accuracy of the trained model, you can use `validate.py` and parse the checkpoint path with `--ckpt_path`.

```
python validate.py -c configs/cmt/cmt_small_ascend.yaml --data_dir /path/to/imagenet --ckpt_path /path/to/ckpt
```

### Deployment

Please refer to the [deployment tutorial](https://mindspore-lab.github.io/mindcv/tutorials/deployment/).

## References

<!--- Guideline: Citation format should follow GB/T 7714. -->
[1] Guo J, Han K, Wu H, et al. Cmt: Convolutional neural networks meet vision transformers[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022: 12175-12185.
