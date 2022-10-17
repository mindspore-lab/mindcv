# SKNet

***

> [SKNet: Selective Kernel Networks](https://arxiv.org/pdf/1903.06586.pdf)

## 模型简介

***

选择性内核网络（SKNet）是由选择性内核卷积堆叠而成，选择性内核卷积的核心思想包括拆分，融合以及选择三个方面。

1. 拆分：使用不同感受野的卷积核对原图进行卷积。举例来说，如果上个分支使用扩张率为1的3X3的空洞卷积，下个分支就使用扩张率为2的3X3的空洞卷积。
2. 融合：将两个分支的经过卷积操作的特征图叠加，然后进行标准的压缩和激励（Squeeze-and-Excitation）操作。
3. 选择：根据选择权重叠加经过不同大小的卷积内核得到的特征图。

## 性能指标

***

|        |         |           |           |    Pynative     |  Pynative  |     Graph      |   Graph    |           |            |
| :----: | ------- | :-------: | :-------: | :-------------: | :--------: | :------------: | :--------: | :-------: | :--------: |
|        | Model   | Top-1 (%) | Top-5 (%) | train (s/epoch) | Infer (ms) | train(s/epoch) | Infer (ms) | Download  |   Config   |
|  GPU   | sknet50 |           |           |                 |            |                |            | [model]() | [config]() |
| Ascend | sknet50 |           |           |                 |            |                |            |           |            |

## 示例

***

### 训练

- 下面是使用预设的yaml配置文件启动训练的示例.

> [configs文件夹](../../configs)中列出了mindcv套件所包含的模型的各个规格的yaml配置文件(在ImageNet数据集上训练和验证的配置)。

  ```shell
  comming soon
  ```

- 下面是使用在ImageNet上预训练的SKNet50模型和Momentum优化器在CIFAR10数据集上进行微调的示例。

  ```shell
  python train.py --model=sknet50 --pretrained --opt=momentum --lr=0.001 dataset=cifar10 --num_classes=10 --dataset_download
  ```

详细的可调参数及其默认值可以在[config.py](../../config.py)中查看。

### 验证

- 下面是使用`validate.py`文件验证SKNet50的预训练模型的精度的示例。

  ```shell
  python validate.py --model=sknet50 --dataset=imagenet --val_split=val --pretrained
  ```

- 下面是使用`validate.py`文件验证SKNet50的自定义参数文件的精度的示例。

  ```shell
  python validate.py --model=sknet50 --dataset=imagenet --val_split=val --ckpt_path='./ckpt/sknet50-best.ckpt'
  ```
