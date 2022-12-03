# RepMLPNet
> [RepMLPNet: Hierarchical Vision MLP with Re-parameterized Locality](https://arxiv.org/pdf/2112.11081v2.pdf)

## 介绍
***

最新版本:https://openaccess.thecvf.com/content/CVPR2022/papers/Ding_RepMLPNet_Hierarchical_Vision_MLP_With_Re-Parameterized_Locality_CVPR_2022_paper.pdf
与旧版本相比，我们不再在传统的ConvNets中使用RepMLP Block作为插件组件。相反，我们用带有分层设计的RepMLP块构建一个MLP体系结构。与MLP- mixer、ResMLP、gMLP、S2-MLP等其他视觉MLP模型相比，RepMLPNet具有良好的性能。
当然，您也可以在模型中使用它作为构建块。
两个版本之间的重叠是结构重新参数化方法(局部注入)，它等价地将conv合并到FC。最新版本的体系结构设计与旧版本(ResNet-50 + RepMLP)有很大不同。

![](repmlpblock.png)

## 性能指标
***

| Model           | Context   |  Top-1 (%)  | Top-5 (%)  |  Params (M)    | Train T. | Infer T. |  Download | Config | Log |
|-----------------|-----------|-------|-------|------------|-------|--------|---|--------|--------------|
| repmlp_t224 | D910x8 | 76.649     |      | 38.3M       | 1011s/epoch | 15.8ms/step | [model]() | [cfg]() | [log]() |


#### 备注

- All models are trained on ImageNet-1K training set and the top-1 accuracy is reported on the validatoin set.
- Context: GPU_TYPE x pieces - G/F, G - graph mode, F - pynative mode with ms function.  

### 训练

- 下面是使用预设的yaml配置文件启动训练的示例.

> [configs文件夹](../../configs)中列出了mindcv套件所包含的模型的各个规格的yaml配置文件(在ImageNet数据集上训练和验证的配置)。

  ```shell
  # train repmlp_t224 on 8 Ascends
  bash ./scripts/run_distribution_ascend.sh ./scripts/rank_table_8pcs.json [DATASET_PATH] ./config/repmlp/repmlp_T224.yaml
  ```


详细的可调参数及其默认值可以在[config.py](../../config.py)中查看。

### 验证

- 下面是使用`validate.py`文件验证shufflenet_v1_g3_x2_0的预训练模型的精度的示例。

  ```shell
  python validate.py --model=RepMLPNet_T224 --data_dir=imagenet_dir --val_split=val --ckpt_path
  ```



