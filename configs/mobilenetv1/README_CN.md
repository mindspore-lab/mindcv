# MobileNetV1

***
> [MobileNetV1: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/pdf/1704.04861.pdf)

## 模型简介

***
与传统卷积神经网络相比，MobileNetV1的参数和计算量在准确率略有降低的前提下大大降低。（与VGG16相比，准确率降低了0.9%，但模型参数仅为VGG的1/32）。该模型是基于简化的架构，使用深度可分离卷积来构建轻量级深度神经网络。同时引入两个简单的全局超参数，可以有效地权衡延迟和准确性。

![](mobilenetv1.png)

## 性能指标
***

| Model           | Context   |  Top-1 (%)  | Top-5 (%)  |  Params (M)    | Train T. | Infer T. |  Download | Config | Log |
|-----------------|-----------|-------------|------------|----------------|----------|----------|-----------|--------|--------------|
| MobileNet_v1_025 | D910x8-G | 54.64       | 78.29      | 0.47           | 113s/epoch |        | [model]() | [cfg]() | [log]() |
| MobileNet_v1_050 | D910x8-G | 66.39       | 86.71      | 1.34           | 120s/epoch |        | [model]() | [cfg]() | [log]() |
| MobileNet_v1_075 | D910x8-G | 70.66       | 89.49      | 2.60           | 128s/epoch |        | [model]() | [cfg]() | [log]() |
| MobileNet_v1_100 | D910x8-G | 71.83       | 90.26      | 4.25           | 130s/epoch |        | [model]() | [cfg]() | [log]() |

#### 备注

- 以上模型均在ImageNet-1K数据集上训练和验证。
- Context: GPU_TYPE x pieces - G/F, G - graph mode, F - pynative mode with ms function.  

## 示例

### 训练

- 下面是使用预设的yaml配置文件启动训练的示例.

> [configs文件夹](../../configs)中列出了mindcv套件所包含的模型的各个规格的yaml配置文件(在ImageNet数据集上训练和验证的配置)。

  ```shell
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  mpirun -n 8 python train.py -c configs/mobilenetv1/mobilenetv1_100_gpu.yaml --data_dir /path/to/imagenet
  ```

  - 下面是使用在ImageNet上预训练的mobilenet_100模型和Momentum优化器在CIFAR10数据集上进行微调的示例。

  ```shell
  python train.py --model=mobilenet_v1_100_224 --pretrained --opt=momentum --lr=0.001 dataset=cifar10 --num_classes=10 --dataset_download
  ```

详细的可调参数及其默认值可以在[config.py](../../config.py)中查看。

### 验证

- 下面是使用`validate.py`文件验证mobilenetv1_100的预训练模型的精度的示例。

  ```shell
  python validate.py --model=mobilenet_v1_100_224 --dataset=imagenet --val_split=val --pretrained
  ```

- 下面是使用`validate.py`文件验证mobilenet_100的自定义参数文件的精度的示例。

  ```shell
  python validate.py --model=mobilenet_v1_100_224 --dataset=imagenet --val_split=val --ckpt_path='./ckpt/mobilenet_v1_100_224-200_2502.ckpt'
  ```

### 部署 (可选)

请参考主页中有关部署的指导
