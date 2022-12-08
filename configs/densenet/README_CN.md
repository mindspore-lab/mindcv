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

| Model       | Context  | Top-1 (%) | Top-5 (%) | Params (M) | Train T.   | Infer T.   | Download  | Config  | Log     |
| ----------- | -------- | --------- | --------- | ---------- | ---------- | ---------- | --------- | ------- | ------- |
| DenseNet121 | D910x8-G | 75.64     | 92.84     | 8.06       | 238s/epoch | 6.7ms/step | [model]() | [cfg]() | [log]() |
| DenseNet161 | D910x8-G | 79.09     | 94.66     | 28.90      | 472s/epoch | 8.7ms/step | [model]() | [cfg]() | [log]() |
| DenseNet169 | D910x8-G | 77.26     | 93.71     | 14.30      | 313s/epoch | 7.4ms/step | [model]() | [cfg]() | [log]() |
| DenseNet201 | D910x8-G | 78.14     | 94.08     | 20.24      | 394s/epoch | 7.9ms/step | [model]() | [cfg]() | [log]() |

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
  mpirun -n 8 python train.py --config configs/densenet/densenet_121_gpu.yaml --data_dir /path/to/imagenet
  ```

详细的可调参数及其默认值可以在[config.py](../../config.py)中查看。

### 验证

- 下面是使用`validate.py`文件验证densenet121的自定义参数文件的精度的示例。

  ```shell
  python validate.py --config configs/densenet/densenet_121_gpu.yaml --data_dir /path/to/imagenet --ckpt_path /path/to/densenet121.ckpt
  ```
