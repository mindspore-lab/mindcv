
# TNT
> [Transformer in Transformer](https://arxiv.org/pdf/2103.00112.pdf)

## Introduction
![122160150-ff1bca80-cea1-11eb-9329-be5031bad78e](https://user-images.githubusercontent.com/41994229/224009923-02ad8d88-1cad-429e-b322-dc80660e8cbd.png)

Illustration of the proposed Transformer-iN-Transformer (TNT) framework. The inner
transformer block is shared in the same layer. The word position encodings are shared across visual
sentences.
## Results

**Implementation and configs for training were taken and adjusted from [this repository](https://gitee.com/cvisionlab/models/tree/tnt/release/research/cv/tnt), which implements tnt model in mindspore.**

Our reproduced model performance on ImageNet-1K is reported as follows.
<div align="center">

| Model    | Context  | Top-1 (%) | Top-5 (%) | Params (M) | Recipe                                                                                        | Download                                                                               |
|----------|----------|-----------|-----------|------------|-----------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------|
| tnt_small | 8xRTX3090 | 74.14     | 92.07     | -       | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/tnt/tnt_s_gpu.yaml) | [weights](https://storage.googleapis.com/huawei-mindspore-hk/TNT/tnt_s_patch16_224_ep138_acc_0.74.ckpt) |
| tnt_small | Converted from PyTorch | 72.51 | 90.68 | - | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/tnt/tnt_s_gpu.yaml) | [weights](https://storage.googleapis.com/huawei-mindspore-hk/TNT/tnt_s_converted_0.718.ckpt) |
| tnt_base | Converted from PyTorch | 79.62 | 94.81 | - | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/tnt/tnt_b_gpu.yaml) | [weights](https://storage.googleapis.com/huawei-mindspore-hk/TNT/tnt_b_converted_0.795.ckpt) |

</div>

#### Notes

- Context: The weights in the table were taken from [official repository](https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/tnt_pytorch) and converted to mindspore format
- Top-1 and Top-5: Accuracy reported on the validation set of ImageNet-1K.

## Quick Start

### Preparation

#### Installation
Please refer to the [installation instruction](https://github.com/mindspore-ecosystem/mindcv#installation) in MindCV.

#### Dataset Preparation
Please download the [ImageNet-1K](https://www.image-net.org/challenges/LSVRC/2012/index.php) dataset for model training and validation.

### Training

* Distributed Training


```shell
# distrubted training on multiple GPU/Ascend devices
mpirun -n 8 python train.py --config configs/tnt/tnt_s_gpu.yaml --data_dir /path/to/imagenet --distributed True
```

> If the script is executed by the root user, the `--allow-run-as-root` parameter must be added to `mpirun`.

Similarly, you can train the model on multiple GPU devices with the above `mpirun` command.

For detailed illustration of all hyper-parameters, please refer to [config.py](https://github.com/mindspore-lab/mindcv/blob/main/config.py).

**Note:**  As the global batch size  (batch_size x num_devices) is an important hyper-parameter, it is recommended to keep the global batch size unchanged for reproduction or adjust the learning rate linearly to a new global batch size.

* Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on a CPU/GPU/Ascend device
python train.py --config configs/tnt/tnt_s_gpu.yaml --data_dir /path/to/dataset --distribute False
```

### Validation

To validate the accuracy of the trained model, you can use `validate.py` and parse the checkpoint path with `--ckpt_path`.

```shell
python validate.py -c configs/tnt/tnt_s_gpu.yaml --data_dir /path/to/imagenet --ckpt_path /path/to/ckpt
```

Or use '--pretrained' parameter to automatically download the checkpoint.

```shell
python validate.py -c configs/tnt/tnt_s_gpu.yaml --data_dir /path/to/imagenet --pretrained
```

### Deployment

Please refer to the [deployment tutorial](https://github.com/mindspore-lab/mindcv/blob/main/tutorials/deployment.md) in MindCV.

## References

Paper - https://arxiv.org/pdf/2103.00112.pdf

Official PyTorch implementation - https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/tnt_pytorch

Official Mindspore implementation - https://gitee.com/cvisionlab/models/tree/tnt/release/research/cv/tnt
