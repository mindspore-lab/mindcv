# ConViT
> [ConViT: Improving Vision Transformers with Soft Convolutional Inductive Biases](https://arxiv.org/abs/2103.10697)

## Introduction

ConViT combines the strengths of convolutional architectures and Vision Transformers (ViTs).
ConViT introduces gated positional self-attention (GPSA), a form of positional self-attention
that can be equipped with a “soft” convolutional inductive bias.
ConViT initializes the GPSA layers to mimic the locality of convolutional layers,
then gives each attention head the freedom to escape locality by adjusting a gating parameter
regulating the attention paid to position versus content information.
ConViT, outperforms the DeiT (Touvron et al., 2020) on ImageNet,
while offering a much improved sample efficiency.[[1](#references)]

<p align="center">
  <img src="https://user-images.githubusercontent.com/52945530/210045403-721c9697-fe7e-429a-bd38-ba244fc8bd1b.png" width=400 />
</p>
<p align="center">
  <em>Figure 1. Architecture of ConViT [<a href="#references">1</a>] </em>
</p>


## Results

Our reproduced model performance on ImageNet-1K is reported as follows.

- ascend 910* with graph mode


<div align="center">


| model       | top-1 (%) | top-5 (%) | params (M) | batch size | cards | ms/step | jit_level | recipe                                                                                           | download                                                                                                |
| ----------- | --------- | --------- | ---------- | ---------- | ----- | ------- | --------- | ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------- |
| convit_tiny | 73.79     | 91.70     | 5.71       | 256        | 8     | 226.51  | O2        | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/convit/convit_tiny_ascend.yaml) | [weights](https://download-mindspore.osinfra.cn/toolkits/mindcv/convit/convit_tiny-1961717e-910v2.ckpt) |

</div>

- ascend 910 with graph mode

<div align="center">


| model       | top-1 (%) | top-5 (%) | params (M) | batch size | cards | ms/step | jit_level | recipe                                                                                           | download                                                                                  |
| ----------- | --------- | --------- | ---------- | ---------- | ----- | ------- | --------- | ------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------- |
| convit_tiny | 73.66     | 91.72     | 5.71       | 256        | 8     | 231.62  | O2        | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/convit/convit_tiny_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/convit/convit_tiny-e31023f2.ckpt) |

</div>

#### Notes
- Top-1 and Top-5: Accuracy reported on the validation set of ImageNet-1K.

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
msrun --bind_core=True --worker_num 8 python train.py --config configs/convit/convit_tiny_ascend.yaml --data_dir /path/to/imagenet
```


For detailed illustration of all hyper-parameters, please refer to [config.py](https://github.com/mindspore-lab/mindcv/blob/main/config.py).

**Note:**  As the global batch size  (batch_size x num_devices) is an important hyper-parameter, it is recommended to keep the global batch size unchanged for reproduction or adjust the learning rate linearly to a new global batch size.

* Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on single NPU device
python train.py --config configs/convit/convit_tiny_ascend.yaml --data_dir /path/to/dataset --distribute False
```

### Validation

To validate the accuracy of the trained model, you can use `validate.py` and parse the checkpoint path with `--ckpt_path`.

```
python validate.py -c configs/convit/convit_tiny_ascend.yaml --data_dir /path/to/imagenet --ckpt_path /path/to/ckpt
```

## References

<!--- Guideline: Citation format should follow GB/T 7714. -->
[1] d’Ascoli S, Touvron H, Leavitt M L, et al. Convit: Improving vision transformers with soft convolutional inductive biases[C]//International Conference on Machine Learning. PMLR, 2021: 2286-2296.
