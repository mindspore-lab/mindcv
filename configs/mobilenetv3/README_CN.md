# MobileNetV3

***
> [MobileNetV3: Searching for MobileNetV3](https://arxiv.org/pdf/1512.00567.pdf)

## 模型简介

***
本文的目标是开发最佳的移动计算机视觉架构，来优化在移动设备上的预测精度与延迟的问题。为了实现这一点，我们引入了：(1)互补搜索技术，(2)适用于移动设备的非线性的模型规格，(3)高效的网络设计，(4)
一个高效的分割解码器。我们对每种技术都进行了大量的实验，并在多种用例和移动电话上验证了它们的有效性。

## 性能指标

***

|        |           |           |           |    Pynative     |  Pynative  |     Graph      |   Graph    |           |            |
| :----: | --------- | :-------: | :-------: | :-------------: | :--------: | :------------: | :--------: | :-------: | :--------: |
|        | Model     | Top-1 (%) | Top-5 (%) | train (s/epoch) | Infer (ms) | train(s/epoch) | Infer (ms) | Download  |   Config   |
|  GPU   | MobileNet_v3_large | 74.56     | 91.79     |                 |            |                |            | [model]() | [config]() |
| Ascend | MobileNet_v3_large | 74.61     | 91.82     |                 |            |                |            |           |            |
|  GPU   | MobileNet_v3_small | 67.46     | 87.07     |                 |            |                |            | [model]() | [config]() |
| Ascend | MobileNet_v3_small | 67.49     | 87.13     |                 |            |                |            |           |            |

## 示例

***

### 训练

- 下面是使用预设的yaml配置文件启动训练的示例.

> [configs文件夹](../../configs)中列出了mindcv套件所包含的模型的各个规格的yaml配置文件(在ImageNet数据集上训练和验证的配置)。

  ```shell
  comming soon
  ```

- 下面是使用在ImageNet上预训练的MobileNetV3_Large_100模型和Adam优化器在CIFAR10数据集上进行微调的示例。

  ```shell
  python train.py --model=mobilenet_v3_large_100 --pretrained --opt=adam --lr=0.001 ataset=cifar10 --num_classes=10 --dataset_download
  ```

详细的可调参数及其默认值可以在[config.py](../../config.py)中查看。

### 验证

- 下面是使用`validate.py`文件验证MobileNetV3_Large_100的预训练模型的精度的示例。

```shell
python validate.py --model=mobilenet_v3_large_100 --dataset=imagenet --val_split=val --pretrained
```

- 下面是使用`validate.py`文件验证MobileNetV3_Large_100的自定义参数文件的精度的示例。

  ```shell
  python validate.py --model=mobilenet_v3_large_100 --dataset=imagenet --val_split=val --ckpt_path='./ckpt/mobilenet_v3_large_100-best.ckpt' 
  ```
