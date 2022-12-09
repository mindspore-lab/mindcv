# ConViT
> [ConViT: Improving Vision Transformers with Soft Convolutional Inductive Biases](https://arxiv.org/pdf/2103.10697.pdf)

## Introduction
***

ConViT combines the strengths of convolutional architectures and Vision Transformers (ViTs). ConViT introduce gated positional self-attention (GPSA), a form of positional self-attention which can be equipped with a “soft” convolutional inductive bias. ConViT initialize the GPSA layers to mimic the locality of convolutional layers, then give each attention head the freedom to escape locality by adjusting a gating parameter regulating the attention paid to position versus content information. ConViT, outperforms the DeiT (Touvron et al., 2020) on ImageNet, while offering a much improved sample efficiency.

![ConViT](convit.png)

## Results
***

| Model            | Context   |  Top-1 (%)  | Top-5 (%)  |  Params (M)    | Train T. | Infer T. |  Download | Config | Log |
|------------------|-----------|-------------|------------|----------------|----------|----------|-----------|--------|-----|
| convit_tiny      | D910x8-G  | 73.66       | 91.72      | 6              | 240s/epoch | 50.7ms/step | [model]() | [cfg]() | [log]() |
| convit_tiny_plus | D910x8-G  | 77.00       | 93.60      | 10             | 247s/epoch | 40.9ms/step | [model]() | [cfg]() | [log]() |
| convit_small     | D910x8-G  | 81.63       | 95.59      | 27             | 490s/epoch | 36.4ms/step | [model]() | [cfg]() | [log]() |
| convit_small_plus| D910x8-G  | 81.82       | 95.41      | 48             | 556s/epoch | 32.7ms/step | [model]() | [cfg]() | [log]() |
| convit_base      | D910x8-G  | 82.10       | 95.52      | 86             | 880s/epoch | 32.8ms/step | [model]() | [cfg]() | [log]() |
| convit_base_plus | D910x8-G  | 82.00       | 95.04      | 152            | 1028s/epoch | 36.6ms/step | [model]() | [cfg]() | [log]() |

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

- **Hyper-parameters.** The hyper-parameter configurations for producing the reported results are stored in the yaml files in `mindcv/configs/convit` folder. For example, to train with one of these configurations, you can run:

  ```shell
  # train convit_tiny on 8 Ascends
  python train.py --config configs/convit/convit_tiny_ascend.yaml --data_dir /path/to/imagenet
  ```

  Note that the number of GPUs/Ascends and batch size will influence the training results. To reproduce the training result at most, it is recommended to use the **same number of GPUs/Ascends** with the same batch size.

Detailed adjustable parameters and their default value can be seen in [config.py](../../config.py).

### Validation

- To validate the model, you can use `validate.py`. Here is an example for convit_tiny to verify the accuracy of your training.

  ```shell
  python validate.py --config configs/convit/convit_tiny_ascend.yaml --data_dir /path/to/imagenet --ckpt_path /path/to/convit_tiny.ckpt
  ```

### Deployment (optional)

Please refer to the deployment tutorial in MindCV.