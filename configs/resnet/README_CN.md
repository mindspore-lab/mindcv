# ResNet

***
> [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)

## 模型简介

***
深层次的神经网络更加难训练，Resnet 是一个残差学习架构，用于简化比以前更深网络的训练，它被明确表述为具有参考层的输入学习残差网络，而不是学习未引用的网络结构。大量的经验证据表明，这些残差网络更容易优化，并且可以从大大增加的深度中获得准确性。

![](resnet.png)

## 性能指标

***

| Model           | Context   |  Top-1 (%)  | Top-5 (%)  |  Params (M)    | Train T. | Infer T. |  Download | Config | Log |
|-----------------|-----------|-------|-------|------------|-------|--------|---|--------|--------------|
| ResNet18 | D910x8-G | 70.10 | 89.58 | 11.70 | 118s/epoch |  | [model]() | [cfg]() | [log]() |
| ResNet34 | D910x8-G | 74.19 | 91.76 | 21.81 | 122s/epoch |  | [model]() | [cfg]() | [log]() |
| ResNet50 | D910x8-G | 76.78 | 93.42 | 25.61 | 213s/epoch |  | [model]() | [cfg]() | [log]() |
| ResNet101 | D910x8-G | 78.06 | 94.15 | 44.65 | 327s/epoch |  | [model]() | [cfg]() | [log]() |
| ResNet152 | D910x8-G | 78.37 | 94.09 | 60.34 | 456s/epoch |  | [model]() | [cfg]() | [log]() |

#### 备注

- All models are trained on ImageNet-1K training set and the top-1 accuracy is reported on the validatoin set.
- Context: GPU_TYPE x pieces - G/F, G - graph mode, F - pynative mode with ms function.  

## 示例

- 以上模型均在ImageNet-1K数据集上训练和验证。
- Context: GPU_TYPE x pieces - G/F, G - graph mode, F - pynative mode with ms function.  

***

### 训练

- 下面是使用预设的yaml配置文件启动训练的示例.

> [configs文件夹](../../configs)中列出了mindcv套件所包含的模型的各个规格的yaml配置文件(在ImageNet数据集上训练和验证的配置)。

  ```shell
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  mpirun -n 8 python train.py -c configs/resnet/resnet_18_gpu.yaml --data_dir /path/to/imagenet
  ```

- 下面是使用在ImageNet上预训练的resnet18模型和Momentum优化器在CIFAR10数据集上进行微调的示例。

  ```shell
  python train.py --model=resnet18 --pretrained --opt=momentum --lr=0.001 dataset=cifar10 --num_classes=10 --dataset_download
  ```

详细的可调参数及其默认值可以在[config.py](../../config.py)中查看。

### 验证

- 下面是使用`validate.py`文件验证resnet18的预训练模型的精度的示例。

  ```shell
  python validate.py --model=resnet18 --dataset=imagenet --val_split=val --pretrained
  ```

- 下面是使用`validate.py`文件验证resnet18的自定义参数文件的精度的示例。

  ```shell
  python validate.py --model=resnet18 --dataset=imagenet --val_split=val --ckpt_path='./ckpt/densenet121-best.ckpt'
  ```
