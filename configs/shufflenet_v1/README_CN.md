# ShuffleNetV1

***
> [ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](https://arxiv.org/pdf/1707.01083.pdf)

## 模型简介

***
ShuffleNet是旷视科技2017年提出的一种计算高效的CNN模型，其和MobileNet和SqueezeNet等一样主要是想应用在移动端。所以，ShuffleNet的设计目标也是如何利用有限的计算资源来达到最好的模型精度，这需要很好地在速度和精度之间做平衡。ShuffleNet的核心是采用了两种操作：pointwise group convolution和channel shuffle，这在保持精度的同时大大降低了模型的计算量。目前移动端CNN模型主要设计思路主要是两个方面：模型结构设计和模型压缩。ShuffleNet和MobileNet一样属于前者，都是通过设计更高效的网络结构来实现模型变小和变快，而不是对一个训练好的大模型做压缩或者迁移。
![](./ShuffleNetV1_Block.png)

## 性能指标

***

| Model                 | Context      | Top-1 (%) | Top-5 (%) | Params (M) | Train T.   | Infer T. | Download  | Config  | Log     |
| --------------------- | ------------ | --------- | --------- | ---------- | ---------- | -------- | --------- | ------- | ------- |
| shufflenet_v1_g3_x0_5 | D910x8-G57.0 | 57.05     | 79.73     | 0.73       | 169s/epoch |          | [model]() | [cfg]() | [log]() |
| shufflenet_v1_g3_x1_0 | D910x8-G     | 67.77     | 87.73     | 1.89       | 192s/epoch |          | [model]() | [cfg]() | [log]() |
| shufflenet_v1_g3_x1_5 | D910x8-G     | 71.53     | 90.17     | 3.48       | 303s/epoch |          | [model]() | [cfg]() | [log]() |
| shufflenet_v1_g3_x2_0 | D910x8-G     | 74.02     | 91.74     | 5.50       | 232s/epoch |          | [model]() | [cfg]() | [log]() |

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
cd mindcv/scripts
bash run_distribute_train_ascend.sh ./hccl_8p_01234567_123.60.231.9.json /tmp/dataset/imagenet ../configs/shufflenet_v1/shufflenet_v1_2.0_ascend.yaml
  ```

详细的可调参数及其默认值可以在[config.py](../../config.py)中查看。

### 验证

- 下面是使用`validate.py`文件验证shufflenet_v1_g3_x2_0的自定义参数文件的精度的示例。

  ```shell
  python validate.py -c configs/shufflenet_v1/shufflenet_v1_2.0_ascend.yaml
  ```


​	
