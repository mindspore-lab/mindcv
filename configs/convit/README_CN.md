# ConViT

> [ConViT: Improving Vision Transformers with Soft Convolutional Inductive Biases](https://arxiv.org/pdf/2103.10697.pdf)

## 模型简介

***

结合了卷积架构和ViTs的优势。 ConViT：用柔性卷积归纳偏差改进视觉Transformer。研究了初始化和归纳偏差在视觉Transformer学习中的重要性。展示了以柔性方式利用卷积约束，融合了架构先验和表达能力的优点。结果是在不增加模型大小、不需要任何调优的情况下，提高训练性和样本有效性的简单方法。引入了门控位置自注意力(GPSA)，一种位置自注意力的形式，可以配备一个"软"卷积归纳偏差。初始化GPSA层以模仿卷积层的位置性，通过调整门控参数来调节对位置信息与内容信息的关注度，给每个关注头以摆脱位置性的自由。由此产生的类似卷积的视觉Transformer架构ConViT在ImageNet上的表现优于DeiT。

![ConViT](convit.png)

## 性能指标

***

| Model            | Context   |  Top-1 (%)  | Top-5 (%)  |  Params (M)    | Train T. | Infer T. |  Download | Config | Log |
|------------------|-----------|-------------|------------|----------------|----------|----------|-----------|--------|-----|
| convit_tiny      | D910x8-G  | 73.66       | 91.72      | 6              | 243s/epoch | 50.7ms/step | [model](https://download.mindspore.cn/toolkits/mindcv/convit/convit_tiny.ckpt) | [cfg](configs/convit/convit_tiny_ascend.yaml) | [log]() |
| convit_tiny_plus | D910x8-G  | 77.00       | 93.60      | 10             | 246s/epoch | 40.9ms/step | [model](https://download.mindspore.cn/toolkits/mindcv/convit/convit_tiny_plus.ckpt) | [cfg](configs/convit/convit_tiny_plus_ascend.yaml) | [log]() |
| convit_small     | D910x8-G  | 81.63       | 95.59      | 27             | 491s/epoch | 36.4ms/step | [model](https://download.mindspore.cn/toolkits/mindcv/convit/convit_small.ckpt) | [cfg](configs/convit/convit_small.yaml) | [log]() |
| convit_small_plus| D910x8-G  | 81.8        | 95.42      | 48             | 557s/epoch | 32.7ms/step | [model](https://download.mindspore.cn/toolkits/mindcv/convit/convit_small_plus.ckpt) | [cfg](configs/convit/convit_small_plus_ascend.yaml) | [log]() |
| convit_base      | D910x8-G  | 82.10       | 95.52      | 86             | 880s/epoch | 32.8ms/step | [model](https://download.mindspore.cn/toolkits/mindcv/convit/convit_base.ckpt) | [cfg](configs/convit/convit_base_ascend.yaml) | [log]() |
| convit_base_plus | D910x8-G  | 81.96       | 95.04      | 152            | 1031s/epoch | 36.6ms/step | [model](https://download.mindspore.cn/toolkits/mindcv/convit/convit_base_plus.ckpt) | [cfg](configs/convit/convit_base_plus_ascend.yaml) | [log]() |



#### 备注

- 以上模型均在ImageNet-1K数据集上训练和验证。
- Context: GPU_TYPE x pieces - G/F, G - graph mode, F - pynative mode with ms function.

## 示例

### 训练

- 下面是使用预设的yaml配置文件启动训练的示例.

> [configs文件夹](../../configs)中列出了mindcv套件所包含的模型的各个规格的yaml配置文件(在ImageNet数据集上训练和验证的配置)。
>
> ```shell
> # train convit_tiny on 8 Ascends
> python train.py --config configs/convit/convit_tiny_ascend.yaml --data_dir /path/to/imagenet
> ```

  详细的可调参数及其默认值可以在[config.py](../../config.py)中查看。

### 验证

- 下面是使用`validate.py`文件验证convit_tiny的预训练模型的精度的示例。

  ```shell
  python validate.py --config configs/convit/convit_tiny_ascend.yaml --data_dir /path/to/imagenet --ckpt_path /path/to/convit_tiny.ckpt
  ```

