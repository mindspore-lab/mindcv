# Res2Net

***

> [Res2Net: A New Multi-scale Backbone Architecture](https://arxiv.org/pdf/1904.01169.pdf)

## 模型简介

***
在本文中，我们通过在一个单个残差模块内构造分层的类残差连接，构建了一种新的CNN模块，我们将该网络命名为Res2Net。Res2Net以更细粒度的方式表示多尺度特征，并且还增加了每个网络层的感受野范围。Res2Net模块还可以融合到目前性能最佳的CNN模型（如ResNet，ResNeXt和DLA）的主干网络中。
我们将这些融合了Res2Net模块的模型在主流的数据集（例如CIFAR-100和ImageNet）上进行了评估，结果表明，相较于基线模型，融合后的模型的性能获得了一致地提升。

![](res2net.png)

## 性能指标

***

|        |                |           |           |    Pynative     |  Pynative  |     Graph      |   Graph    |           |            |
| :----: | -------------- | :-------: | :-------: | :-------------: | :--------: | :------------: | :--------: | :-------: | :--------: |
|        | Model          | Top-1 (%) | Top-5 (%) | train (s/epoch) | Infer (ms) | train(s/epoch) | Infer (ms) | Download  |   Config   |
|  GPU   | res2net50      |           |           |                 |            |                |            | [model]() | [config]() |
| Ascend | res2net50      |           |           |                 |            |                |            |           |            |
|  GPU   | res2net101     |           |           |                 |            |                |            | [model]() | [config]() |
| Ascend | res2net101     |           |           |                 |            |                |            |           |            |
|  GPU   | res2net50_v1b  |           |           |                 |            |                |            | [model]() | [config]() |
| Ascend | res2net50_v1b  |           |           |                 |            |                |            |           |            |
|  GPU   | res2net101_v1b |           |           |                 |            |                |            | [model]() | [config]() |
| Ascend | res2net101_v1b |           |           |                 |            |                |            |           |            |

## 示例

***

### 训练

- 下面是使用预设的yaml配置文件启动训练的示例.

> [configs文件夹](../../configs)中列出了mindcv套件所包含的模型的各个规格的yaml配置文件(在ImageNet数据集上训练和验证的配置)。

  ```shell
  comming soon
  ```

- 下面是使用在ImageNet上预训练的Res2Net50模型和Momentum优化器在CIFAR10数据集上进行微调的示例。

  ```shell
  python train.py --model=res2net50 --pretrained --opt=momentum --lr=0.001 dataset=cifar10 --num_classes=10 --dataset_download
  ```

详细的可调参数及其默认值可以在[config.py](../../config.py)中查看。

### 验证

- 下面是使用`validate.py`文件验证Res2Net50的预训练模型的精度的示例。

  ```shell
  python validate.py --model=res2net50 --dataset=imagenet --val_split=val --pretrained
  ```

- 下面是使用`validate.py`文件验证Res2Net50的自定义参数文件的精度的示例。

  ```shell
  python validate.py --model=res2net50 --dataset=imagenet --val_split=val --ckpt_path='./ckpt/res2net50-best.ckpt'
  ```
