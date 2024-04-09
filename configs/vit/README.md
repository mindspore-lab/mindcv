# ViT
<!--- Guideline: use url linked to abstract in ArXiv instead of PDF for fast loading.  -->

> [ An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)

## Introduction
<!--- Guideline: Introduce the model and architectures. Cite if you use/adopt paper explanation from others. -->

Vision Transformer (ViT) achieves remarkable results compared to convolutional neural networks (CNN) while obtaining fewer computational resources for pre-training. In comparison to convolutional neural networks (CNN), Vision Transformer (ViT) shows a generally weaker inductive bias resulting in increased reliance on model regularization or data augmentation (AugReg) when training on smaller datasets.

The ViT is a visual model based on the architecture of a transformer originally designed for text-based tasks, as shown in the below figure. The ViT model represents an input image as a series of image patches, like the series of word embeddings used when using transformers to text, and directly predicts class labels for the image. ViT exhibits an extraordinary performance when trained on enough data, breaking the performance of a similar state-of-art CNN with 4x fewer computational resources. [[2](#references)]

<!--- Guideline: If an architecture table/figure is available in the paper, put one here and cite for intuitive illustration. -->

<p align="center">
  <img src="https://user-images.githubusercontent.com/8156835/210041797-6576b2f4-3d77-41d9-b5f0-16fed3f261d8.png" width=800 />
</p>
<p align="center">
  <em> Figure 1. Architecture of ViT [<a href="#references">1</a>] </em>
</p>

## Results
<!--- Guideline:
Table Format:
- Model: model name in lower case with _ seperator.
- Context: Training context denoted as {device}x{pieces}-{MS mode}, where mindspore mode can be G - graph mode or F - pynative mode with ms function. For example, D910x8-G is for training on 8 pieces of Ascend 910 NPU using graph mode.
- Top-1 and Top-5: Keep 2 digits after the decimal point.
- Params (M): # of model parameters in millions (10^6). Keep 2 digits after the decimal point
- Recipe: Training recipe/configuration linked to a yaml config file. Use absolute url path.
- Download: url of the pretrained model weights. Use absolute url path.
-->

Our reproduced model performance on ImageNet-1K is reported as follows.

<div align="center">

| Model        | Context  | Top-1 (%) | Top-5 (%) | Params (M) | Recipe                                                                                        | Download                                                                                |
|--------------|----------|--|-----------|------------|-----------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|
| vit_b_32_224 | D910x8-G | 77.45 | 93.27     | 87.46      | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/vit/vit_b32_224_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/vit/vit_b_32_224-4a1c9d8e.ckpt) |
| vit_l_16_224 | D910x8-G | 81.25 | 95.53     | 303.31     | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/vit/vit_l16_224_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/vit/vit_l_16_224-d2635f8b.ckpt) |
| vit_l_32_224 | D910x8-G | 74.57 | 91.01     | 305.52     | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/vit/vit_l32_224_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/vit/vit_l_32_224-8c8ea164.ckpt) |

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
<!--- Guideline: Avoid using shell script in the command line. Python script preferred. -->

* Distributed Training

It is easy to reproduce the reported results with the pre-defined training recipe. For distributed training on multiple Ascend 910 devices, please run

```shell
# distributed training on multiple GPU/Ascend devices
mpirun -n 8 python train.py --config configs/vit/vit_b32_224_ascend.yaml --data_dir /path/to/imagenet
```
> If the script is executed by the root user, the `--allow-run-as-root` parameter must be added to `mpirun`.

Similarly, you can train the model on multiple GPU devices with the above `mpirun` command.

For detailed illustration of all hyper-parameters, please refer to [config.py](https://github.com/mindspore-lab/mindcv/blob/main/config.py).

**Note:**
1) As the global batch size  (batch_size x num_devices) is an important hyper-parameter, it is recommended to keep the global batch size unchanged for reproduction or adjust the learning rate linearly to a new global batch size.
2) The current configuration with a batch_size of 512, was initially set for a machine with 64GB of VRAM. To avoid running out of memory (OOM) on machines with smaller VRAM, consider reducing the batch_size to 256 or lower.

* Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on a CPU/GPU/Ascend device
python train.py --config configs/vit/vit_b32_224_ascend.yaml --data_dir /path/to/dataset --distribute False
```

### Validation

To validate the accuracy of the trained model, you can use `validate.py` and parse the checkpoint path with `--ckpt_path`.

```
python validate.py -c configs/vit/vit_b32_224_ascend.yaml --data_dir /path/to/imagenet --ckpt_path /path/to/ckpt
```

### Deployment

To deploy online inference services with the trained model efficiently, please refer to the [deployment tutorial](https://mindspore-lab.github.io/mindcv/tutorials/deployment/).

## References
<!--- Guideline: Citation format GB/T 7714 is suggested. -->

[1] Dosovitskiy A, Beyer L, Kolesnikov A, et al. An image is worth 16x16 words: Transformers for image recognition at scale[J]. arXiv preprint arXiv:2010.11929, 2020.

[2] "Vision Transformers (ViT) in Image Recognition – 2022 Guide", https://viso.ai/deep-learning/vision-transformer-vit/
