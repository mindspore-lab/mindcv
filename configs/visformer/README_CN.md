# Visformer

***
> [Visformer: The Vision-friendly Transformer](https://arxiv.org/pdf/2104.12533.pdf)

## 模型简介

***
在过去的几年中，将Transformer 应用于视觉问题的研究快速发展。虽然一些研究人员已经证明了基于Transformer的模型具有良好的数据拟合能力，仍有越来越多的
证据表明，这些模型会过拟合，特别是训练数据有限时。本文逐步将基于Transformer的模型转化为基于卷积的模型，在转化过程中获得的结果能够提供有用信息来提升
视觉识别效果。基于此，我们提出了一个命名为Visformer的新结构，它是"Vision-friendly Transformer"的缩写。
![](visformer.png)

## 性能指标

***

| Model            | Context   |  Top-1 (%)  | Top-5 (%)  |  Params (M)    | Train T.   | Infer T. |  Download | Config | Log |
|------------------|-----------|-------------|------------|----------------|------------|----------|-----------|--------|-----|
| visformer_tiny   | D910x8-G  | 78.28       | 94.15      | 10             | 496/epoch  | 300.7ms/step | [model](https://download.mindspore.cn/toolkits/mindcv/visformer/) | [cfg]() | [log]() |
| visformer_tiny2  | D910x8-G  | 78.82       | 94.41      | 9              | 390s/epoch | 602.5ms/step | [model](https://download.mindspore.cn/toolkits/mindcv/visformer/) | [cfg]() | [log]() |
| visformer_small  | D910x8-G  | 81.73       | 95.88      | 40             | 445s/epoch | 155.9ms/step | [model](https://download.mindspore.cn/toolkits/mindcv/visformer/) | [cfg]() | [log]() |
| visformer_small2 | D910x8-G  | 82.17       | 95.90      | 23             | 440s/epoch | 153.1ms/step | [model](https://download.mindspore.cn/toolkits/mindcv/visformer/) | [cfg]() | [log]() |
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
  mpirun -n 8 python train.py --config configs/visformer/visformer_tiny_ascend.yaml --data_dir /path/to/imagenet
  ```

详细的可调参数及其默认值可以在[config.py](../../config.py)中查看。

### 验证

- 下面是使用`validate.py`文件验证densenet121的自定义参数文件的精度的示例。

  ```shell
  python validate.py --config configs/visformer/visformer_tiny_ascend.yaml --data_dir /path/to/imagenet --ckpt_path /path/to/visformer_tiny.ckpt
  ```
