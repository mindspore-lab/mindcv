# MobileNetV1

***

> [MobileNetV1: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/pdf/1704.04861.pdf)

## 模型简介

***
相较于传统的卷积神经网络，MobileNetV1的主要优势是在模型精度略有降低的前提下大大减少了模型的参数量和训练的计算量。（比如，MobileNetV1的参数量只有VGG16的1/32，但是精度只降低了0.9%）。MobileNetV1模型是基于流线型架构，使用深度可分离卷积来构建轻量级深度神经网络。另外，模型还引入了两个简单的全局超参数，使得模型可以在延迟和准确率之间做折中，以此来适应不同的应用场景。

## 性能指标

***

|        |           |           |           |    Pynative     |  Pynative  |     Graph      |   Graph    |           |            |
| :----: | --------- | :-------: | :-------: | :-------------: | :--------: | :------------: | :--------: | :-------: | :--------: |
|        | Model     | Top-1 (%) | Top-5 (%) | train (s/epoch) | Infer (ms) | train(s/epoch) | Infer (ms) | Download  |   Config   |
|  GPU   | MobileNet_v1_100 | 71.95     | 90.41     |                 |            |                |            | [model]() | [config]() |
| Ascend | MobileNet_v1_100 | 71.83     | 90.26     |                 |            |                |            |           |            |
|  GPU   | MobileNet_v1_075 | 70.84     | 89.63     |                 |            |                |            | [model]() | [config]() |
| Ascend | MobileNet_v1_075 | 70.66     | 89.49     |                 |            |                |            |           |            |
|  GPU   | MobileNet_v1_050 | 66.37     | 86.71     |                 |            |                |            | [model]() | [config]() |
| Ascend | MobileNet_v1_050 | 66.39     | 86.85     |                 |            |                |            |           |            |
|  GPU   | MobileNet_v1_025 | 54.58     | 78.27     |                 |            |                |            | [model]() | [config]() |
| Ascend | MobileNet_v1_025 | 54.64     | 78.29     |                 |            |                |            |           |            |

## 示例

***

### 训练

- 下面是使用预设的yaml配置文件启动训练的示例.

> [configs文件夹](../../configs)中列出了mindcv套件所包含的模型的各个规格的yaml配置文件(在ImageNet数据集上训练和验证的配置)。

  ```shell
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  mpirun --allow-run-as-root -n 8 python train.py -c configs/mobilenetv1/mobilenetv1_075_gpu.yaml
  ```

- 下面是使用在ImageNet上预训练的InceptionV3模型和Adam优化器在CIFAR10数据集上进行微调的示例。

  ```shell
  python train.py --model=mobilenet_v1_075_224 --pretrained --opt=adam --lr=0.001 dataset=cifar10 --num_classes=10 --dataset_download
  ```

详细的可调参数及其默认值可以在[config.py](../../config.py)中查看。

### 验证

- 下面是使用`validate.py`文件验证MobileNetV1_075_224的预训练模型的精度的示例。

  ```shell
  python validate.py --model=mobilenet_v1_075_224 --dataset=imagenet --val_split=val --pretrained
  ```

- 下面是使用`validate.py`文件验证MobileNetV1_075_224的自定义参数文件的精度的示例。

  ```shell
  python validate.py --model=mobilenet_v1_075_224 --dataset=imagenet --val_split=val --ckpt_path='./ckpt/mobilenet_v1_075_224-best.ckpt' 
  ```
