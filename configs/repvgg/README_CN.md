# RepVGG

***
> [RepVGG: Making VGG-style ConvNets Great Again](https://arxiv.org/pdf/2101.03697.pdf)

## 模型简介

***
RepVGG是一种类似VGG架构的网络，但是其性能优于许多复杂的模型。RepVGG具有以下优点：

1）该模型具有类似VGG的朴素的拓扑结构，没有任何分支。也就是说模型的每一层的输入都只是上一层的输出，其输出又作为下一层的输入。

2）该模型的主干网络只有3X3的卷积层和ReLU激活函数两种结构。

3）模型的实际架构设计过程中也没有使用自动搜索，手动优化，复合缩减或其他繁琐的技术。

## 性能指标

***

|        |           |           |           |    Pynative     |  Pynative  |     Graph      |   Graph    |           |            |
| :----: | --------- | :-------: | :-------: | :-------------: | :--------: | :------------: | :--------: | :-------: | :--------: |
|        | Model     | Top-1 (%) | Top-5 (%) | train (s/epoch) | Infer (ms) | train(s/epoch) | Infer (ms) | Download  |   Config   |
|  GPU   | RepVGG_A0 | 71.98     | 90.36     |                 |            |                |            | [model]() | [config]() |
| Ascend | RepVGG_A0 | 71.87     | 90.43     |                 |            |                |            |           |            |

## 示例

***

### 训练

- 下面是使用预设的yaml配置文件启动训练的示例.

> [configs文件夹](../../configs)中列出了mindcv套件所包含的模型的各个规格的yaml配置文件(在ImageNet数据集上训练和验证的配置)。

  ```shell
  comming soon
  ```

- 下面是使用在ImageNet上预训练的RepVGG_A0模型和Adam优化器在CIFAR10数据集上进行微调的示例。

  ```shell                                                                                                                                                                                                                                                                                                                                                                              
  python train.py --model=RepVGG_A0 --pretrained --opt=adam --lr=0.001 ataset=cifar10 --num_classes=10 --dataset_download
  ```

详细的可调参数及其默认值可以在[config.py](../../config.py)中查看。

### 验证

- 下面是使用`validate.py`文件验证RepVGG_A0的预训练模型的精度的示例。

  ```shell
  python validate.py --model=RepVGG_A0 --dataset=imagenet --val_split=val --pretrained
  ```

- 下面是使用`validate.py`文件验证RepVGG_A0的自定义参数文件的精度的示例。

  ```shell
  python validate.py --model=RepVGG_A0 --dataset=imagenet --val_split=val --ckpt_path='./ckpt/RepVGG_A0-best.ckpt' 
  ```
