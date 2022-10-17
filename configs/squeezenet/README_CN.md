# SqueezeNet

***
> [SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size](https://arxiv.org/pdf/1602.07360.pdf)

## 模型简介

***
SqueezeNet主要由Fire模块组成，是一种体量较小的卷积神经网路。它在ImageNet数据集上的准确率与AlexNet相当，但是参数量仅有AlexNet参数量的1/50。SqueezeNet至少具备一下三种优势：（1）体量较小的卷积网络在分布式训练过程中需要跨服务器之间需要进行更少的通信。（2）体量较小的卷积网络需要更少的带宽,
将一个新的模型从云端导出到自动汽车。（3）体量较小的卷积网络提供了在FPGA等其他内存有限的硬件上部署的可行性。此外, 使用模型压缩技术, 我们可以将SqueezeNet压缩到小于0.5MB（比AlexNet小510倍）。
下图展示了SqueezeNet的宏观架构，左边的图展示的是SqueezeNet的基准架构，中间的图表示的是在某些Fire模块之间进行简单的旁路链接的SqueezeNet，右边的图表示的是在Fire模块之间使用复杂的旁路链接的SqueezeNet。

![](squeezenet.png)

## 性能指标

***

|        |              |           |           |    Pynative     |  Pynative  |     Graph      |   Graph    |           |            |
| :----: | ------------ | :-------: | :-------: | :-------------: | :--------: | :------------: | :--------: | :-------: | :--------: |
|        | Model        | Top-1 (%) | Top-5 (%) | train (s/epoch) | Infer (ms) | train(s/epoch) | Infer (ms) | Download  |   Config   |
| GPU | squeezenet_1.0 | 59.48 | 81.22 |  |  |  |  | [model]() | [config]() |
| Ascend | squeezenet_1.0 |   59.49   | 81.22 |  |  |  |  |  |  |
|  GPU   | squeezenet_1.1 | 58.99 | 80.98 |                 |            |                |            | [model]() | [config]() |
| Ascend | squeezenet_1.1 | 58.99 |   80.99   |                 |            |                |            |           |            |

## 示例

***

### 训练

- 下面是使用预设的yaml配置文件启动训练的示例.

> [configs文件夹](../../configs)中列出了mindcv套件所包含的模型的各个规格的yaml配置文件(在ImageNet数据集上训练和验证的配置)。

  ```shell
  coming soon
  ```

- 下面是使用在ImageNet上预训练的SqueezeNet1_0模型和Momentum优化器在CIFAR10数据集上进行微调的示例。

  ```shell
  python train.py --model=squeezenet1_0 --pretrained --opt=momentum --lr=0.001 dataset=cifar10 --num_classes=10 --dataset_download
  ```

详细的可调参数及其默认值可以在[config.py](../../config.py)中查看。

### 验证

- 下面是使用`validate.py`文件验证SqueezeNet1_0的预训练模型的精度的示例。

  ```shell
  python validate.py --model=squeezenet1_0 --dataset=imagenet --val_split=val --pretrained
  ```

- 下面是使用`validate.py`文件验证SqueezeNet1_0的自定义参数文件的精度的示例。

  ```shell
  python validate.py --model=squeezenet1_0 --dataset=imagenet --val_split=val --ckpt_path='./ckpt/squeezenet1_0-best.ckpt'
  ```
