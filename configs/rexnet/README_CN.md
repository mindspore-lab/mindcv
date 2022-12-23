# ReXNet

***
> [ReXNet: Rethinking Channel Dimensions for Efficient Model Design](https://arxiv.org/abs/2007.00992)

## 模型简介

***
这是网络体系结构设计的一种新范式。ReXNet提出了一套设计原则来解决现有网络中的表征瓶颈问题。Rexnet将这些设计原则与现有的网络单元结合起来，得到了一个新的网络Rexnet，它实现了极大的性能改进。



## 性能指标


| Model           | Context   |  Top-1 (%)  | Top-5 (%)  |  Params (M)    | Train T. | Infer T. |  Download | Config | Log |
|-----------------|-----------|-------|-------|------------|-------|--------|---|--------|--------------|
| rexnet_x09 | D910x8-G | 77.07 | 93.41    |      |   |   | [model](https://download.mindspore.cn/toolkits/mindcv/rexnet/)  | [cfg]() | [log]() |
| rexnet_x10 | D910x8-G | 77.38 | 93.60    |       |   |   | [model](https://download.mindspore.cn/toolkits/mindcv/rexnet/)  | [cfg]() | [log]() |
| rexnet_x13 | D910x8-G | 79.06 | 94.28 |  |   |   | [model](https://download.mindspore.cn/toolkits/mindcv/rexnet/)  | [cfg]() | [log]() |
| rexnet_x15 | D910x8-G | 79.94 | 94.74  |   |   |   | [model](https://download.mindspore.cn/toolkits/mindcv/rexnet/)  | [cfg]() | [log]() |
| rexnet_x20 | D910x8-G | 80.6 | 94.99  |   |   |   | [model](https://download.mindspore.cn/toolkits/mindcv/rexnet/)  | [cfg]() | [log]() |

#### Notes

- All models are trained on ImageNet-1K training set and the top-1 accuracy is reported on the validatoin set.
- Context: GPU_TYPE x pieces - G/F, G - graph mode, F - pynative mode with ms function.  

## 示例

### 训练

- 下面是使用预设的yaml配置文件启动训练的示例.

> [configs文件夹](../../configs)中列出了mindcv套件所包含的模型的各个规格的yaml配置文件(在ImageNet数据集上训练和验证的配置)。

  ```shell
python train.py --config ./config/rexnet/rexnet_x10.yaml
  ```

  - 下面是使用Adam优化器在CIFAR10数据集上对预训练的rexnet x1.0进行微调的示例。

  ```shell
python train.py --model=rexnet_x10 --pretrained --opt=momentum --lr=0.001 dataset=cifar10 --num_classes=10 --dataset_download
  ```

详细的可调参数及其默认值可以在[config.py](../../config.py)中查看。

### 验证

- 下面是使用`validate.py`文件验证rexnet_x1.0的预训练模型的精度的示例。

  ```shell
  python validate.py --model=rexnet_x10 --dataset=imagenet --val_split=val --pretrained
  ```

- 下面是使用`validate.py`文件验证rexnet_x1.0的自定义参数文件的精度的示例。

  ```shell
  python validate.py --model=rexnet_x10 --dataset=imagenet --val_split=val --ckpt_path='./rexnetx10_ckpt/rexnet-best.ckpt'
  ```

### 部署 (可选)

请参考主页中有关部署的指导
