# ShuffleNetV2

***
> [ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://arxiv.org/pdf/1807.11164.pdf)

## 模型简介

***
ShuffleNetV2中提出了一个关键点，之前的轻量级网络都是通过计算网络复杂度的一个间接度量，即FLOPs为指导。通过计算浮点运算量来描述轻量级网络的快慢。但是从来不直接考虑运行的速度。在移动设备中的运行速度不仅仅需要考虑FLOPs，还需要考虑其他的因素，比如内存访问成本(memory accesscost)和平台特点(platform characterics)。

所以，根据这两个原则，ShuffleNetV2提出了四种有效的网络设计原则：

- 当卷积层的输入特征矩阵与输出特征矩阵channel相等时MAC最小(保持FLOPs不变时)；
- 当GConv的groups增大时(保持FLOPs不变时)，MAC也会增大；
- 网络设计的碎片化程度越高，速度越慢；
- Element-wise操作带来的影响是不可忽视的。
  ![](./ShuffleNetV2_Block.png)

## 性能指标

***

| Model              | Context  | Top-1 (%) | Top-5 (%) | Params (M) | Train T.   | Infer T. | Download  | Config  | Log     |
| ------------------ | -------- | --------- | --------- | ---------- | ---------- | -------- | --------- | ------- | ------- |
| shufflenet_v2_x0_5 | D910x8-G | 60.68     | 82.44     | 1.37       | 99s/epoch  |          | [model]() | [cfg]() | [log]() |
| shufflenet_v2_x1_0 | D910x8-G | 69.51     | 88.67     | 2.29       | 101s/epoch |          | [model]() | [cfg]() | [log]() |
| shufflenet_v2_x1_5 | D910x8-G | 72.59     | 90.79     | 3.53       | 125s/epoch |          | [model]() | [cfg]() | [log]() |
| shufflenet_v2_x2_0 | D910x8-G | 75.14     | 92.13     | 7.44       | 149s/epoch |          | [model]() | [cfg]() | [log]() |

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
mpirun -n 8 python train.py -c configs/shufflenetv2/shufflenet_v2_x2_0.yaml --data_dir /path/to/imagenet
  ```

- 下面是使用在ImageNet上预训练的shufflenet_v2_x2_0模型和Momentum优化器在CIFAR10数据集上进行微调的示例。

  ```shell
  python train.py --model=shufflenet_v2_x2_0 --pretrained --opt=momentum --lr=0.001 dataset=cifar10 --num_classes=10 --dataset_download
  ```

详细的可调参数及其默认值可以在[config.py](../../config.py)中查看。

### 验证

- 下面是使用`validate.py`文件验证shufflenet_v2_x2_0的预训练模型的精度的示例。

  ```shell
  python validate.py --model=shufflenet_v2_x2_0 --dataset=imagenet --val_split=val --pretrained
  ```

- 下面是使用`validate.py`文件验证shufflenet_v2_x2_0的自定义参数文件的精度的示例。

  ```shell
  python validate.py --model=shufflenet_v2_x2_0 --dataset=imagenet --val_split=val --ckpt_path='./ckpt/shufflenet_v2_x2_0-best.ckpt'
  ```
