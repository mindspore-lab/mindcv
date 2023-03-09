# XCiT: Cross-Covariance Image Transformers

> [XCiT: Cross-Covariance Image Transformers](https://arxiv.org/abs/2106.09681)

## 介绍

XCiT 模型提出了一种自注意力的“转置”版本，它跨特征通道进行操作，而不是对token进行操作，其中的交互是基于keys和queries之间的交叉协方差矩阵。 由此产生的交叉协方差注意力 (XCA) 在tokens数量上具有线性复杂性，并允许高效处理高分辨率图像。 我们的交叉协方差图像变换器 (XCiT) 基于 XCA 构建，结合了传统transformers的准确性和卷积架构的可扩展性。

<p align="center">
  <img src="https://user-images.githubusercontent.com/51260139/211969416-b57b3aff-49b0-4048-b970-55d9196ed63b.png" width=600 />
</p>
<p align="center">
  <em>Figure 1. XCiT模型结构 [<a href="#references">1</a>] </em>
</p>



* **时间和内存上的线性复杂度**

  XCiT 模型具有关于patches/tokens数量的线性复杂度: $\mathcal{O}(N d ^2)$


| ![](https://user-images.githubusercontent.com/51260139/211969388-0658c89b-c41c-4df9-a295-5b3431b626b7.png) | ![](https://user-images.githubusercontent.com/51260139/211969950-92b15d1d-0b08-4075-9a12-faf40cd49efa.png) |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
|                      Peak Memory (推理)                      |                   Millisecond/Image (推理)                   |


## 结果

我们在 ImageNet-1K 上的再现模型性能报告如下。

<div align="center">

| 模型        | Context  | Top-1 (%) | Top-5 (%) | 参数量 (M) | 训练策略                                                                                        | 权重下载                                                                       |
|--------------|----------|-----------|-----------|------------|-----------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------|
| xcit_tiny_12_p16 | D910x8-G | 77.674     | 93.792      | 7      | [yaml](https://github.com/bnucsy/mindcv/tree/main/configs/xcit/xcit_tiny_12_p16_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/xcit/xcit_tiny_12_p16_best.ckpt) |

</div>


#### 注意

* Context：训练上下文表示为 {device}x{pieces}-{MS mode}，其中 mode 可以是 G - graph mode 或 F - pynative mode。 例如，D910x8-G 用于使用graph mode在 8 块 Ascend 910 NPU 上进行训练。

- Top-1 and Top-5: 在 ImageNet-1K 的验证集上报告的准确性。

## 快速开始

### 准备工作

#### 安装

请参考MindCV的 [安装教程](https://github.com/mindspore-lab/mindcv#installation)。

#### 数据集准备

请下载 [ImageNet-1K](https://www.image-net.org/challenges/LSVRC/2012/index.php) 数据集进行训练和验证。

### 模型训练

* 分布式训练

使用预设的训练策略很容易复现结果。 在多台Ascend 910设备分布式训练，请运行

```shell
# 在多GPU或多Ascend设备上分布式训练
mpirun -n 8 python train.py --config configs/xcit/xcit_tiny_12_p16_ascend.yaml --data_dir /path/to/imagenet
```

> 如果脚本由 root 用户执行，则必须将 `--allow-run-as-root` 参数添加到 `mpirun`。

同样，您可以使用上述 `mpirun` 命令在多个 GPU 设备上训练模型。

所有超参数的详细说明请参考[config.py](https://github.com/mindspore-lab/mindcv/blob/main/config.py)。

**注意：** 由于全局批量大小（batch_size x num_devices）是一个重要的超参数，建议保持全局批量大小不变以进行复制或将学习率线性调整为新的全局批量大小。

* 单卡训练

如果您想在没有分布式训练的情况下在较小的数据集上训练或微调模型，请运行：

```shell
# 在单个CPU/GPU/Ascend设备上训练
python train.py --config configs/xcit/xcit_tiny_12_p16_ascend.yaml --data_dir /path/to/dataset --distribute False
```

### 模型验证

要验证训练模型的准确性，您可以使用 `validate.py` 并使用` --ckpt_path` 设置checkpoint文件路径。

```
python validate.py -c configs/xcit/xcit_tiny_12_p16_ascend.yaml --data_dir /path/to/imagenet --ckpt_path /path/to/ckpt
```

### 模型部署

请参考MindCV的 [部署教程](https://github.com/mindspore-lab/mindcv/blob/main/tutorials/deployment.md)。

## 参考文献

<!--- Guideline: Citation format should follow GB/T 7714. -->
[1] Ali A, Touvron H, Caron M, et al. Xcit: Cross-covariance image transformers[J]. Advances in neural information processing systems, 2021, 34: 20014-20027.
