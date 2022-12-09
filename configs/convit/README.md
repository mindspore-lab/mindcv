# ConViT
> [ConViT: Improving Vision Transformers with Soft Convolutional Inductive Biases](https://arxiv.org/pdf/2103.10697.pdf)

## Introduction
***

ConViT结合了卷积架构和ViT的优势。ConViT引入了门控位置自注意力（GPSA），这是一种位置自注意力的形式，可以配备“软”卷积归纳偏置。ConViT初始化GPSA层以模拟卷积层的局部性，然后通过调整调节对位置与内容信息的注意力的门控参数，让每个注意力头可以自由地逃离局部性。由此产生的类似卷积的ViT架构ConViT在ImageNet上优于DeiT(Touvron et al., 2020)，同时提供了大大提高的样本效率。

![ConViT](convit.png)

## 性能指标
***

| Model            | Context   |  Top-1 (%)  | Top-5 (%)  |  Params (M)    | Train T. | Infer T. |  Download | Config | Log |
|------------------|-----------|-------------|------------|----------------|----------|----------|-----------|--------|-----|
| convit_tiny      | D910x8-G  | 73.66       | 91.72      | 6              | 240s/epoch | 50.7ms/step | [model]() | [cfg]() | [log]() |
| convit_tiny_plus | D910x8-G  | 77.00       | 93.60      | 10             | 247s/epoch | 40.9ms/step | [model]() | [cfg]() | [log]() |
| convit_small     | D910x8-G  | 81.63       | 95.59      | 27             | 490s/epoch | 36.4ms/step | [model]() | [cfg]() | [log]() |
| convit_small_plus| D910x8-G  | 81.82       | 95.41      | 48             | 556s/epoch | 32.7ms/step | [model]() | [cfg]() | [log]() |
| convit_base      | D910x8-G  | 82.10       | 95.52      | 86             | 880s/epoch | 32.8ms/step | [model]() | [cfg]() | [log]() |
| convit_base_plus | D910x8-G  | 82.00       | 95.04      | 152            | 1028s/epoch | 36.6ms/step | [model]() | [cfg]() | [log]() |

#### 备注

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