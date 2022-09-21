import sys
sys.path.append('.')

from mindcv.utils.download import DownLoad
import os
import matplotlib.pyplot as plt
import numpy as np
from mindcv.models import create_model

from mindcv.loss import create_loss
from mindcv.optim import create_optimizer
from mindcv.scheduler import create_scheduler
from mindspore import Model, LossMonitor, TimeMonitor #, CheckpointConfig, ModelCheckpoint


freeze_backbone = False
visualize = False

dataset_url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/intermediate/Canidae_data.zip"
root_dir = "./"

if not os.path.exists(os.path.join(root_dir, 'data/Canidae')):
    DownLoad().download_and_extract_archive(dataset_url, root_dir)
    
from mindcv.data import create_dataset, create_transforms, create_loader

num_workers = 8

# 数据集目录路径
data_dir = "./data/Canidae/"

# 加载自定义数据集
dataset_train = create_dataset(root=data_dir, split='train', num_parallel_workers=num_workers)
dataset_val = create_dataset(root=data_dir, split='val', num_parallel_workers=num_workers)

# 定义和获取数据处理及增强操作
trans_train = create_transforms(dataset_name='ImageNet', is_training=True)
trans_val = create_transforms(dataset_name='ImageNet',is_training=False)

# 
loader_train = create_loader(
        dataset=dataset_train,
        batch_size=16,
        is_training=True,
        num_classes=2,
        transform=trans_train,
        num_parallel_workers=num_workers,
    )
    

loader_val = create_loader(
        dataset=dataset_val,
        batch_size=5,
        is_training=True,
        num_classes=2,
        transform=trans_val,
        num_parallel_workers=num_workers,
    )

images, labels= next(loader_train.create_tuple_iterator())
#images = data["image"]
#labels = data["label"]

print("Tensor of image", images.shape)
print("Labels:", labels)

# class_name对应label，按文件夹字符串从小到大的顺序标记label
class_name = {0: "dogs", 1: "wolves"}

if visualize:
    plt.figure(figsize=(15, 7))
    for i in range(len(labels)):
        # 获取图像及其对应的label
        data_image = images[i].asnumpy()
        data_label = labels[i]
        # 处理图像供展示使用
        data_image = np.transpose(data_image, (1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        data_image = std * data_image + mean
        data_image = np.clip(data_image, 0, 1)
        # 显示图像
        plt.subplot(3, 6, i + 1)
        plt.imshow(data_image)
        plt.title(class_name[int(labels[i].asnumpy())])
        plt.axis("off")

    plt.show()

network = create_model(model_name='densenet121', num_classes=2, pretrained=True)


# 定义优化器和损失函数
lr = 1e-3 if freeze_backbone else 1e-4
opt = create_optimizer(network.trainable_params(), opt='adam', lr=lr) 
loss = create_loss(name='CE')

if freeze_backbone:
    # freeze backbone
    for param in network.get_parameters():
        if param.name not in ["classifier.weight", "classifier.bias"]:
            param.requires_grad = False

            
# 实例化模型
model = Model(network, loss_fn=loss, optimizer=opt, metrics={'accuracy'}) 
print('Training...')
model.train(10, loader_train, callbacks=[LossMonitor(5), TimeMonitor(5)], dataset_sink_mode=False)
print('Evaluating...')
res = model.eval(loader_val)
print(res)