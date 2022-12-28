# ViT

> [ An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)

## Introduction
***

Vision Transformer (ViT) achieves remarkable results compared to convolutional neural networks (CNN) while obtaining fewer computational resources for pre-training. In comparison to convolutional neural networks (CNN), Vision Transformer (ViT) show a generally weaker inductive bias resulting in increased reliance on model regularization or data augmentation (AugReg) when training on smaller datasets. 

The ViT is a visual model based on the architecture of a transformer originally designed for text-based tasks, as shown in the below figure. The ViT model represents an input image as a series of image patches, like the series of word embeddings used when using transformers to text, and directly predicts class labels for the image. ViT exhibits an extraordinary performance when trained on enough data, breaking the performance of a similar state-of-art CNN with 4x fewer computational resources.

![vit](./vit.png)

## Results

Our reproduced model performance on ImageNet-1K is reported as follows.

***

| Model           | Context   |  Top-1 (%) | Top-5 (%)  |  Params (M) | Recipe  | Download |
|-----------------|-----------|------------|------------|-------------|---------|----------|
| vit_b_32_224 | D910x8-G | 75.86  | 92.08    | 87.46    | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/vit/vit_b32_224_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/vit/vit_b_32_224.ckpt)  |
| vit_l_16_224 | D910x8-G | 76.34  | 92.79    | 303.32    | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/vit/vit_l16_224_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/vit/vit_l_16_224.ckpt)  |
| vit_l_32_224 | D910x8-G | 73.71  | 90.92    | 303.326    | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/vit/vit_b32_224_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/vit/vit_l_32_224.ckpt)  |

#### Notes
- Context: Training context denoted as {device}x{pieces}-{MS mode}, where mindspore mode can be G - graph mode or F - pynative mode with ms function. For example, D910x8-G is for training on 8 pieces of Ascend 910 NPU using graph mode. 
- Top-1 and Top-5: Accuracy reported on the validatoin set of ImageNet-1K. 


## Quick Start
***
### Preparation

#### Installation
Please refer to the [installation instruction](https://github.com/mindspore-lab/mindcv#installation) in MindCV.

#### Dataset Preparation
Please download the [ImageNet-1K](https://www.image-net.org/challenges/LSVRC/2012/index.php) dataset for model training and validation.

### Training

* Disitrubuted Training
It is easy to reproduce the reported results with the pre-defined training recipe. For distributed training on multiple Ascend 910 devices, please run

```shell
# distrubted training on multiple GPU/Ascend devices
mpirun -n 8 python train.py --config configs/vit/vit_b32_224_ascend.yaml --data_dir /path/to/imagenet
```
  
Similary, you can run the above command to train your model on GPU devices.

Note: As global batch size is an important hyper-parameter in model training. it is recommended to use the same global batch size (batch_size x num_devices) for reproduction or adjust the learning rate linearly for a different global batch size.

Detailed adjustable parameters and their default value can be seen in [config.py](https://github.com/mindspore-lab/mindcv/blob/main/config.py).


* Standalone Training
If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on a CPU/GPU/Ascend device
python train.py --config configs/vit/vit_b32_224_ascend.yaml --data_dir /path/to/dataset --distribute False
```

### Validation

- To validate the trained model, you can use `validate.py` and specify the path of the trained model in `--ckpt_path`.

```
python validate.py -c configs/vit/vit_b32_224_ascend.yaml --data_dir /path/to/imagenet --ckpt_path /path/to/ckpt
```

### Deployment

Please refer to the [deployment tutorial](https://github.com/mindspore-lab/mindcv/blob/main/tutorials/deployment.md) in MindCV.

