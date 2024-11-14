# ViT

<!--- Guideline: use url linked to abstract in ArXiv instead of PDF for fast loading.  -->

> [ An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)



## Introduction

<!--- Guideline: Introduce the model and architectures. Cite if you use/adopt paper explanation from others. -->

Vision Transformer (ViT) achieves remarkable results compared to convolutional neural networks (CNN) while obtaining
fewer computational resources for pre-training. In comparison to convolutional neural networks (CNN), Vision
Transformer (ViT) shows a generally weaker inductive bias resulting in increased reliance on model regularization or
data augmentation (AugReg) when training on smaller datasets.

The ViT is a visual model based on the architecture of a transformer originally designed for text-based tasks, as shown
in the below figure. The ViT model represents an input image as a series of image patches, like the series of word
embeddings used when using transformers to text, and directly predicts class labels for the image. ViT exhibits an
extraordinary performance when trained on enough data, breaking the performance of a similar state-of-art CNN with 4x
fewer computational resources. [[2](#references)]

<!--- Guideline: If an architecture table/figure is available in the paper, put one here and cite for intuitive illustration. -->

<p align="center">
  <img src="https://user-images.githubusercontent.com/8156835/210041797-6576b2f4-3d77-41d9-b5f0-16fed3f261d8.png" width=800 />
</p>
<p align="center">
  <em> Figure 1. Architecture of ViT [<a href="#references">1</a>] </em>
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
msrun --bind_core=True --worker_num 8 python train.py --config configs/vit/vit_b32_224_ascend.yaml --data_dir /path/to/imagenet
```



For detailed illustration of all hyper-parameters, please refer
to [config.py](https://github.com/mindspore-lab/mindcv/blob/main/config.py).

**Note:**

1) As the global batch size  (batch_size x num_devices) is an important hyper-parameter, it is recommended to keep the
   global batch size unchanged for reproduction or adjust the learning rate linearly to a new global batch size.
2) The current configuration with a batch_size of 512, was initially set for a machine with 64GB of VRAM. To avoid
   running out of memory (OOM) on machines with smaller VRAM, consider reducing the batch_size to 256 or lower.
   Simultaneously, to maintain the consistency of training results, please scale the learning rate down proportionally
   with decreasing batch_size.

* Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on single NPU device
python train.py --config configs/vit/vit_b32_224_ascend.yaml --data_dir /path/to/dataset --distribute False
```

### Validation

To validate the accuracy of the trained model, you can use `validate.py` and parse the checkpoint path
with `--ckpt_path`.

```
python validate.py -c configs/vit/vit_b32_224_ascend.yaml --data_dir /path/to/imagenet --ckpt_path /path/to/ckpt
```

## Performance

Our reproduced model performance on ImageNet-1K is reported as follows.

Experiments are tested on ascend 910* with mindspore 2.3.1 graph mode.

*coming soon*

Experiments are tested on ascend 910 with mindspore 2.3.1 graph mode.

*coming soon*

### Notes

- top-1 and top-5: Accuracy reported on the validation set of ImageNet-1K.

## References

<!--- Guideline: Citation format GB/T 7714 is suggested. -->

[1] Dosovitskiy A, Beyer L, Kolesnikov A, et al. An image is worth 16x16 words: Transformers for image recognition at
scale[J]. arXiv preprint arXiv:2010.11929, 2020.

[2] "Vision Transformers (ViT) in Image Recognition – 2022 Guide", https://viso.ai/deep-learning/vision-transformer-vit/
