# MobileNetV2

***
> [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/pdf/1801.04381.pdf)

## 模型简介

***
该模型是一种新的神经网络体系结构，它专门为移动和资源受限的环境量身定制。
此网络推动了移动定制计算机视觉模型的最先进水平，在保持相同精度的同时，显著减少了所需的操作和内存数量。

该模型的主要创新点是提出了一个新的层模块:The Inverted Residual with Linear Bottleneck。该模块以一个低维压缩表示作为输入，该表示首先扩展到高维，然后用轻量级深度卷积进行滤波。
随后，利用线性卷积将特征投影回低维表示

![](mobilenetv2.png)

## 性能指标
***

| Model           | Context   |  Top-1 (%)  | Top-5 (%)  |  Params (M)    | Train T. | Infer T. |  Download | Config | Log |
|-----------------|-----------|-------------|------------|----------------|----------|----------|-----------|--------|--------------|
| MobileNet_v2_075 | D910x8-G | 69.76       | 89.28      | 2.66           | 106s/epoch |        | [model]() | [cfg]() | [log]() |
| MobileNet_v2_100 | D910x8-G | 72.02       | 90.92      | 3.54           | 98s/epoch |        | [model]() | [cfg]() | [log]() |
| MobileNet_v2_140 | D910x8-G | 74.97       | 92.32      | 6.15           | 157s/epoch |        | [model]() | [cfg]() | [log]() |

#### 备注

- 以上模型均在ImageNet-1K数据集上训练和验证。
- Context: GPU_TYPE x pieces - G/F, G - graph mode, F - pynative mode with ms function.  

## 示例

### 训练

- 下面是使用预设的yaml配置文件启动训练的示例.

> [configs文件夹](../../configs)中列出了mindcv套件所包含的模型的各个规格的yaml配置文件(在ImageNet数据集上训练和验证的配置)。

  ```shell
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  mpirun -n 8 python train.py -c configs/mobilenetv2/mobilenetv2_100_gpu.yaml --data_dir /path/to/imagenet
  ```

  - 下面是使用在ImageNet上预训练的mobilenet_100模型和Momentum优化器在CIFAR10数据集上进行微调的示例。

  ```shell
  python train.py -c configs/mobilenetv2/mobilenetv2_100_gpu.yaml --data_dir /path/to/imagenet
  ```

详细的可调参数及其默认值可以在[config.py](../../config.py)中查看。

### 验证

- 下面是使用`validate.py`文件验证mobilenetv2_100的预训练模型的精度的示例。

  ```shell
  python validate.py -c /path/to/val.yaml --data_dir /path/to/imagenet
  ```

- 下面是使用`validate.py`文件验证mobilenet_100的自定义参数文件的精度的示例。

  ```shell
  python validate.py -c /path/to/val.yaml --data_dir /path/to/imagenet --dataset=imagenet --val_split=val --ckpt_path='./ckpt/mobilenet_v2_100_224-200_625.ckpt'
  ```

### 部署 (可选)

请参考主页中有关部署的指导
