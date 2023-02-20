# 特性

- **高易用性** MindCV将视觉框架分解为各种可配置组件，方便您使用MindCV定制您的数据管道、模型和学习管道。

```python
>>> import mindcv
# 创建一个数据集
>>> dataset = mindcv.create_dataset('cifar10', download=True)
# 创建一个模型
>>> network = mindcv.create_model('resnet50', pretrained=True)
```

用户可以在一个命令行中自定义和启动他们的迁移学习或训练任务。

```shell
# 仅使用一个命令行即可启动迁移学习任务
python train.py --model swin_tiny --pretrained --opt adamw --lr 0.001 --data_dir = {data_dir}
```

- **业内最佳** MindCV提供了大量包括SwinTransformer在内的基于CNN和基于Transformer结构的视觉模型。同时，还提供了它们的预训练权重以及性能测试报告，帮助用户正确地选择和使用他们所需要的模型。

- **灵活高效** MindCV是基于新一代高效的深度学习框架MindSpore编写的，可以运行在多种硬件平台上（CPU/GPU/Ascend），还同时支持高效的图模式和灵活的调试模式。
