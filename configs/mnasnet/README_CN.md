# MnasNet

***
> [MnasNet: Platform-Aware Neural Architecture Search for Mobile](https://arxiv.org/abs/1807.11626)

## 模型简介

***
为移动设备设计卷积神经网络(CNN)是一个挑战，因为移动模型需要小而快，但仍然准确。尽管在设计和改进移动cnn方面已经付出了大量的努力，但当有这么多的架构可能性需要考虑时，手动平衡这些权衡是非常困难的。在本文中，我们提出了一种自动移动神经体系结构搜索(MNAS)方法，该方法显式地将模型延迟纳入主要目标，以便搜索可以识别一个在精度和延迟之间实现良好权衡的模型。与之前的工作不同，延迟是通过另一个通常不准确的代理(如FLOPS)来考虑的，我们的方法通过在手机上执行模型直接测量真实世界的推断延迟。为了进一步在灵活性和搜索空间大小之间取得平衡，我们提出了一种新的可分解层次搜索空间，以鼓励整个网络的层次多样性。
![](mnasnet.png)

## 性能指标

***

|        |              |           |           |    Pynative     |  Pynative  |     Graph      |   Graph    |           |            |
| :----: | ------------ | :-------: | :-------: | :-------------: | :--------: | :------------: | :--------: | :-------: | :--------: |
|        | Model        | Top-1 (%) | Top-5 (%) | train (s/epoch) | Infer (ms) | train(s/epoch) | Infer (ms) | Download  |   Config   |
| GPU | MnasNet-B1-0_75 | 72.15 | 90.53 |  |  |  |  |  |  |
| Ascend | MnasNet-B1-0_75 | 71.81 | 90.53 |  |  | 96 |  | [model]() | [config]() |
|  GPU   | MnasNet-B1-1_0 | 74.31 | 91.89 |                 |            |                |            |  |  |
| Ascend | MnasNet-B1-1_0 | 74.28 | 91.70 |                 |            | 96 |            | [model]() | [config]() |
| GPU | MnasNet-B1-1_0 |           |  | | | | |  |  |
| Ascend | MnasNet-B1-1_4 | 76.01 | 92.83 | | | 121 | | [model]() | [config]() |

## 示例

***

### 训练

- 下面是使用预设的yaml配置文件启动训练的示例.

> [configs文件夹](../../configs)中列出了mindcv套件所包含的模型的各个规格的yaml配置文件(在ImageNet数据集上训练和验证的配置)。

  ```shell
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  mpirun -n 8 python train.py -c configs/mnasnet/mnasnet0.75_gpu.yaml --data_dir /path/to/imagenet
  ```

详细的可调参数及其默认值可以在[config.py](../../config.py)中查看。

### 验证

- 下面是使用`validate.py`文件验证mnasnet0_75的预训练模型的精度的示例。

  ```shell
  python validate.py 
  -c configs/mnasnet/mnasnet0.75_ascend.yaml 
  --data_dir=/path/to/imagenet 
  --ckpt_path=/path/to/ckpt
  ```

- 下面是使用`validate.py`文件验证mnasnet0_75的自定义参数文件的精度的示例。

  ```shell
  python validate.py 
  -c configs/mnasnet/mnasnet0.75_ascend.yaml 
  --data_dir=/path/to/imagenet 
  --ckpt_path=/path/to/ckpt
  ```
