# PoolFormer
> [MetaFormer Is Actually What You Need for Vision](https://arxiv.org/pdf/2111.11418v3.pdf)

## 介绍
***


这项工作的目标不是设计复杂的令牌混合器来实现SOTA性能，而是证明Transformer模型的能力主要来自通用体系结构MetaFormer。Pooling/PoolFormer只是支持我们声明的工具。
![](metaformer.png)

图1:在ImageNet-1K验证集上的MetaFormer和基于MetaFormer模型的性能。我们认为，Transformer/ mlp类模型的能力主要来自通用架构MetaFormer，而不是配备的特定令牌混合器。为了演示这一点，我们利用一个非常简单的非参数操作符pooling来进行非常基本的令牌混合。令人惊讶的是，结果模型PoolFormer的性能始终优于DeiT和ResMLP，如(b)所示，这很好地支持了MetaFormer实际上是我们实现竞争性性能所需要的。(b)中的RSB-ResNet表示结果来自“ResNet Strikes Back”，其中ResNet用改进的训练程序训练了300个epoch。

![](poolformer.png)
图2:(a) PoolFormer的总体框架。(b) PoolFormer块的架构。与Transformer块相比，它用一个极其简单的非参数操作符pooling代替了注意力，只进行基本的令牌混合。

## 性能指标
***


| Model           | Context   |  Top-1 (%)  | Top-5 (%)  |  Params (M)    | Train T. | Infer T. |  Download | Config | Log |
|-----------------|-----------|-------|-------|------------|-------|--------|---|--------|--------------|
| poolformer_s12 | D910x8 | 77.094     |   93.394   |  12       | 396.24s/epoch | 19.9ms/step | [model]() | [cfg]() | [log]() |


#### 备注

- All models are trained on ImageNet-1K training set and the top-1 accuracy is reported on the validatoin set.
- Context: GPU_TYPE x pieces - G/F, G - graph mode, F - pynative mode with ms function.  

### 训练

- 下面是使用预设的yaml配置文件启动训练的示例.

> [configs文件夹](../../configs)中列出了mindcv套件所包含的模型的各个规格的yaml配置文件(在ImageNet数据集上训练和验证的配置)。

  ```shell
  # train poolformer_s12 on 8 Ascends
  bash ./scripts/run_distribution_ascend.sh ./scripts/rank_table_8pcs.json [DATASET_PATH] ./config/poolformer/poolformer_s12.yaml
  ```


详细的可调参数及其默认值可以在[config.py](../../config.py)中查看。

### 验证

- 下面是使用`validate.py`文件验证shufflenet_v1_g3_x2_0的预训练模型的精度的示例。

  ```shell
  python validate.py --model=poolformer_s12 --data_dir=imagenet_dir --val_split=val --ckpt_path
  ```



