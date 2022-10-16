# GoogleNet
***
> [GoogleNet: Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842)

##  模型简介
***
GoogLeNet 是 Christian Szegedy 在 2014 年提出的一种新的深度学习网络。在此之前，AlexNet、VGG 等网络通过增加网络的深度（层数）来达到更好的训练效果，但是随着层数的增加，越来越多的问题开始出现，比如过拟合、梯度消失、梯度爆炸等。Inception结构的提出希望从另一个角度来提升了训练效果，该结构可以帮助我们更高效地利用计算资源，在同样的计算量下可以提取更多的特征。
![](GoogLeNet网络.jpg)

## 性能指标


|        |           |           |           |    Pynative     |  Pynative  |     Graph      |   Graph    |           |            |
| :----: | --------- | :-------: | :-------: | :-------------: | :--------: | :------------: | :--------: | :-------: | :--------: |
|        | Model     | Top-1 (%) | Top-5 (%) | train (s/epoch) | Infer (ms) | train(s/epoch) | Infer (ms) | Download  |   Config   |
|  GPU   | googlenet |           |           |     260.898     |            |    260.434     |            | [model]() | [config]() |
| Ascend | googlenet |           |           |                 |            |                |            |           |            |

## 示例

***

### 训练

- 下面是使用预设的yaml配置文件启动训练的示例.

> [configs文件夹](../../configs)中列出了mindcv套件所包含的模型的各个规格的yaml配置文件(在ImageNet数据集上训练和验证的配置)。

  ```shell
  comming soon
  ```

- 下面是使用在ImageNet上预训练的GoogleNet模型和Momentum优化器在CIFAR10数据集上进行微调的示例。

  ```shell
  python train.py --model=googlenet --pretrained --opt=momentum --lr=0.001 dataset=cifar10 --num_classes=10 --dataset_download
  ```

详细的可调参数及其默认值可以在[config.py](../../config.py)中查看。

### 验证

- 下面是使用`validate.py`文件验证GoogleNet的预训练模型的精度的示例。
  ```shell
  python validate.py --model=googlenet --dataset=imagenet --val_split=val --pretrained
  ```

- 下面是使用`validate.py`文件验证GoogleNet的自定义参数文件的精度的示例。

  ```shell
  python validate.py --model=googlenet --dataset=imagenet --val_split=val --ckpt_path='./ckpt/googlenet-best.ckpt'
  ```

