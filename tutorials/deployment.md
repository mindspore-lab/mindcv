# 部署推理服务

本文以Densenet121网络为例，演示基于MindSpore Serving进行部署推理服务的方法。

MindSpore Serving是一个轻量级、高性能的推理服务模块，旨在帮助MindSpore开发者在生产环境中高效部署在线推理服务。当用户使用MindSpore完成模型训练后，导出MindSpore模型，即可使用MindSpore Serving创建该模型的推理服务。



## 环境准备
进行部署前，需确保已经正确安装了MindSpore Serving，并配置了环境变量。MindSpore Serving安装和配置可以参考[MindSpore Serving安装页面](https://www.mindspore.cn/serving/docs/zh-CN/master/serving_install.html)。

## 模型导出

实现跨平台或硬件执行推理（如昇腾AI处理器、MindSpore端侧、GPU等），需要通过网络定义和CheckPoint生成MindIR格式模型文件。在Mindspore中，网络模型导出的函数为`export`，主要参数如下所示：

- `net`：MindSpore网络结构。
- `inputs`：网络的输入，支持输入类型为Tensor。当输入有多个时，需要一起传入，如`ms.export(network, ms.Tensor(input1), ms.Tensor(input2), file_name='network', file_format='MINDIR')`。
- `file_name`：导出模型的文件名称，如果`file_name`没有包含对应的后缀名(如.mindir)，设置`file_format`后系统会为文件名自动添加后缀。
- `file_format`：MindSpore目前支持导出”AIR”，”ONNX”和”MINDIR”格式的模型。

下面代码以Densenet121为例，导出mindcv的预训练网络模型，获得MindIR格式模型文件。


```python
from mindcv.models import create_model
import numpy as np
import mindspore as ms

model = create_model(model_name='densenet121', num_classes=1000, pretrained=True)

input_np = np.random.uniform(0.0, 1.0, size=[1, 3, 224, 224]).astype(np.float32)

# 导出文件densenet121_224.mindir到当前文件夹
ms.export(model, ms.Tensor(input_np), file_name='densenet121_224', file_format='MINDIR')
```

## 部署Serving推理服务

### 配置服务
启动Serving服务，执行本教程需要如下文件列表:
```Text
demo
├── densenet121
│   ├── 1
│   │   └── densenet121_224.mindir
│   └── servable_config.py
│── serving_server.py
├── serving_client.py
└── test_image
    ├─ dog
    │   ├─ dog.jpg
    │   └─ ……
    └─ ……
```


- `densenet121`为模型文件夹，文件夹名即为模型名。
- `densenet121_224.mindir`为上一步网络生成的模型文件，放置在文件夹1下，1为版本号，不同的版本放置在不同的文件夹下，版本号需以纯数字串命名，默认配置下启动最大数值的版本号的模型文件。
- `servable_config.py`为模型配置脚本，对模型进行声明、入参和出参定义。
- `serving_server.py`为启动服务脚本文件。
- `serving_client.py`为启动客户端脚本文件。
- `test_image`中为测试图片。

其中，模型配置文件`serving_config.py`内容如下：
```python
from mindspore_serving.server import register

# 进行模型声明，其中declare_model入参model_file指示模型的文件名称，model_format指示模型的模型类别
model = register.declare_model(model_file="densenet121_224.mindir", model_format="MindIR")

# Servable方法的入参由Python方法的入参指定，Servable方法的出参由register_method的output_names指定
@register.register_method(output_names=["score"])
def predict(image):
    x = register.add_stage(model, image, outputs_count=1)
    return x
```

### 启动服务

Mindspore的`server`函数提供两种服务部署，一种是gRPC方式，一种是通过RESTful方式，本教程以gRPC方式为例。服务启动脚本`serving_server.py`把本地目录下的densenet121部署到设备0，并启动地址为127.0.0.1:5500的gRPC服务器。脚本文件内容如下：
```python
import os
import sys
from mindspore_serving import server

def start():
    servable_dir = os.path.dirname(os.path.realpath(sys.argv[0]))

    servable_config = server.ServableStartConfig(servable_directory=servable_dir, servable_name="densenet121",
                                                 device_ids=0)
    server.start_servables(servable_configs=servable_config)
    server.start_grpc_server(address="127.0.0.1:5500")

if __name__ == "__main__":
    start()
```

当服务端打印如下日志时，表示Serving gRPC服务启动成功。

```text
Serving gRPC server start success, listening on 127.0.0.1:5500
```

### 执行推理
使用`serving_client.py`，启动Python客户端。客户端脚本使用`mindcv.data`的`create_transforms`, `create_dataset`和`create_loader`函数，进行图片预处理，再传送给Serving服务器。对服务器返回的结果进行后处理，打印图片的预测标签。
```python
import os
from mindspore_serving.client import Client
import numpy as np
from mindcv.data import create_transforms, create_dataset, create_loader

num_workers = 1

# 数据集目录路径
data_dir = "./test_image/"

dataset = create_dataset(root=data_dir, split='', num_parallel_workers=num_workers)
transforms_list = create_transforms(dataset_name='ImageNet', is_training=False)
data_loader = create_loader(
        dataset=dataset,
        batch_size=1,
        is_training=False,
        num_classes=1000,
        transform=transforms_list,
        num_parallel_workers=num_workers
    )
with open("imagenet1000_clsidx_to_labels.txt") as f:
    idx2label = eval(f.read())

def postprocess(score):
    max_idx = np.argmax(score)
    return idx2label[max_idx]

def predict():
    client = Client("127.0.0.1:5500", "densenet121", "predict") 
    instances = []
    images, _ = next(data_loader.create_tuple_iterator())
    image_np = images.asnumpy().squeeze()   
    instances.append({"image": image_np})
    result = client.infer(instances)

    for instance in result:
        label = postprocess(instance["score"])
        print(label)

if __name__ == '__main__':
    predict()
```

执行后显示如下返回值，说明Serving服务已正确执行Densenet121网络模型的推理。
```text
Labrador retriever
```
