# BigTransfer

***
> [Big Transfer (BiT): General Visual Representation Learning](https://arxiv.org/abs/1912.11370)

## 模型简介

***
在为视觉任务训练深度神经网络时，预训练模型的迁移提高了训练并简化了超参数调整。我们重新审视了在大型监督数据集上进行预训练并在目标任务上进行微调的范例。我们扩大了预训练的规模，并提出了一个简单的方法，我们称之为 Big Transfer (BiT)。通过组合一些精心挑选的组件，并使用简单的启发式迁移，已经在20多个数据集上有了很好的表现。BiT在各种大小的数据集上都表现很好，从每类1个示例到100万个示例。BiT 在 ILSVRC-2012 上达到87.5%的top-1 精度，在CIFAR-10上达到99.4%，在包含19个任务的VTAB数据集上
上达到76.3%。在小型数据集上，BiT在每类 10 个示例的ILSVRC-2012上达到 76.8%的准确率，在每类 10 个示例的CIFAR-10上达到 97.0%。我们对导致高迁移性能的主要组件进行了详细分析。

![BiT](./BiT.png)

## 性能指标

***

|    Model     | Context  | Top-1 (%) | Top-5 (%) | Params(M) |  Train T.  |  Infer T.   |                           Download                           |                            Config                            |                             Log                              |
| :----------: | :------: | :-------: | :-------: | :-------: | :--------: | :---------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|    BiT50-S   | D910x8-G |   76.81   |   93.17   |    25     | 652s/epoch | 189.8ms/step | [model](https://download.mindspore.cn/toolkits/mindcv/bit/BiTresnet50.ckpt) | [cfg](https://github.com/mindspore-lab/mindcv/blob/main/configs/BigTransfer/BiT50_ascend.yaml) | [log](https://github.com/mindspore-lab/mindcv/tree/main/configs/BigTransfer) |

## 备注

- 以上模型均在ImageNet-1K数据集上训练和验证。
- Context: D910 x 8 - G, D910 - 昇腾910, x8 - 8卡, G - 静态图模型.

***

### 训练

- 下面是使用预设的yaml配置文件启动训练的示例.

> [configs文件夹](../../configs)中列出了mindcv套件所包含的模型的各个规格的yaml配置文件(在ImageNet数据集上训练和验证的配置)。

  ```shell
mpirun -n 8 python train.py -c configs/BigTransfer/BiT50_ascend.yaml --data_dir /path/to/imagenet
  ```

详细的可调参数及其默认值可以在[config.py](../../config.py)中查看。

### 验证

- 下面是使用`validate.py`文件验证BiT-50的自定义参数文件的精度的示例。

  ```shell
  python validate.py -c configs/BigTransfer/BiT50_ascend.yaml --data_dir /path/to/imagenet --ckpt_path /path/to/ckpt
  ```
