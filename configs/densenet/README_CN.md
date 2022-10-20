# DenseNet

***
> [Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993.pdf)

## 模型简介

***
最近的研究表明如果将接近输入的层和接近输出的层之间短接，卷积神经网络可以更深、精度更高并且更加高效。我们利用这个观察结果提出了密集卷积网络(DenseNet)，它的每一层在前向反馈过程中都和后面的所有层有连接。与$L$层传统卷积神经网络(
每层的输出都只作为其相邻的后一层的输入）有$L$个连接不同，DenseNet的每一层的输入包括该层之前所有层的输出，同时该层的输出又作为之后所有层的输入，因此$L$层的DenseNet有$L(L+1)
/2$个连接关系。DenseNets有几个非常具有竞争力的优势：缓解了梯度消失，加强了特征传播，增强了特征复用以及显著地减少了参数量。
![](densenet.png)

## 性能指标

***

|        |              |           |           |    Pynative     |  Pynative  |     Graph      |   Graph    |           |            |
| :----: | ------------ | :-------: | :-------: | :-------------: | :--------: | :------------: | :--------: | :-------: | :--------: |
|        | Model        | Top-1 (%) | Top-5 (%) | train (s/epoch) | Infer (ms) | train(s/epoch) | Infer (ms) | Download  |   Config   |
| GPU | DenseNet121 | 75.60 | 92.73 |  |  |  |  | [model]() | [config]() |
| Ascend | DenseNet121 | 75.60 | 92.73 |  |  |  |  |  |  |
|  GPU   | DenseNet161 | 79.10 | 94.65 |                 |            |                |            | [model]() | [config]() |
| Ascend | DenseNet161  | 79.10 | 94.64 |                 |            |                |            |           |            |
| GPU | DenseNet169 | 76.38 | 93.34 | | | | | [model]() | [config]() |
| Ascend | DenseNet169 | 76.37 | 93.33 | | | | | | |
| GPU | DenseNet201 | 78.08 | 94.13 | | | | | [model]() | [config]() |
| Ascend | DenseNet201 | 78.08 | 94.12 | | | | | | |

## 示例

***

### 训练

- 下面是使用预设的yaml配置文件启动训练的示例.

> [configs文件夹](../../configs)中列出了mindcv套件所包含的模型的各个规格的yaml配置文件(在ImageNet数据集上训练和验证的配置)。

  ```shell
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  mpirun -n 8 python train.py -c configs/densenet/densenet_121_gpu.yaml --data_dir /path/to/imagenet
  ```

- 下面是使用在ImageNet上预训练的densenet121模型和Momentum优化器在CIFAR10数据集上进行微调的示例。

  ```shell
  python train.py --model=densenet121 --pretrained --opt=momentum --lr=0.001 dataset=cifar10 --num_classes=10 --dataset_download
  ```

详细的可调参数及其默认值可以在[config.py](../../config.py)中查看。

### 验证

- 下面是使用`validate.py`文件验证densenet121的预训练模型的精度的示例。

  ```shell
  python validate.py --model=densenet121 --dataset=imagenet --val_split=val --pretrained
  ```

- 下面是使用`validate.py`文件验证densenet121的自定义参数文件的精度的示例。

  ```shell
  python validate.py --model=densenet121 --dataset=imagenet --val_split=val --ckpt_path='./ckpt/densenet121-best.ckpt'
  ```
