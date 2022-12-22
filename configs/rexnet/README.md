# ReXNet
***
> [ReXNet: Rethinking Channel Dimensions for Efficient Model Design](https://arxiv.org/abs/2007.00992)

##  Introduction
***
This is a new paradigm for network architecture design. ReXNet proposes a set of design principles to solve the Representational Bottleneck problem in the existing network. Rexnet combines these design principles with the existing network units to obtain a new network, RexNet, which achieves a great performance improvement.



## Results
***

|        |           |           |           |    Pynative     |  Pynative  |     Graph      |   Graph    |           |            |
| :----: | --------- | :-------: | :-------: | :-------------: | :--------: | :------------: | :--------: | :-------: | :--------: |
|        | Model     | Top-1 (%) | Top-5 (%) | train (s/epoch) | Infer (ms) | train(s/epoch) | Infer (ms) | Download  |   Config   |
| Ascend | rexnet_x09 | 77.07 | 93.41 |          |            |         |            | [model](https://download.mindspore.cn/toolkits/mindcv/rexnet/) | [cfg](https://github.com/mindspore-lab/mindcv/tree/main/configs/rexnet) |
| Ascend | rexnet_x10 | 77.38 | 93.60 |          |            |         |            | [model](https://download.mindspore.cn/toolkits/mindcv/rexnet/) | [cfg](https://github.com/mindspore-lab/mindcv/tree/main/configs/rexnet) |
| Ascend | rexnet_x13 | 79.06 | 94.28 |          |            |         |            | [model](https://download.mindspore.cn/toolkits/mindcv/rexnet/) | [cfg](https://github.com/mindspore-lab/mindcv/tree/main/configs/rexnet) |
| Ascend | rexnet_x15 | 79.94 | 94.74 |          |            |         |            | [model](https://download.mindspore.cn/toolkits/mindcv/rexnet/) | [cfg](https://github.com/mindspore-lab/mindcv/tree/main/configs/rexnet) |
| Ascend | rexnet_x20 | 80.6 | 94.99 |          |            |         |            | [model](https://download.mindspore.cn/toolkits/mindcv/rexnet/) | [cfg](https://github.com/mindspore-lab/mindcv/tree/main/configs/rexnet) |



## Quick Start

***

### Train

- The [yaml config files](../../configs) that yield competitive results on ImageNet for different models are listed in the `configs` folder. To trigger training using preset yaml config. 

  ```shell
  python train.py --config ./config/rexnet/rexnet_x10.yaml
  ```


- Here is the example for finetuning a pretrained rexnet x1.0 on CIFAR10 dataset using Adam optimizer.

  ```shell
  python train.py --model=rexnet_x10 --pretrained --opt=momentum --lr=0.001 dataset=cifar10 --num_classes=10 --dataset_download
  ```

Detailed adjustable parameters and their default value can be seen in [config.py](../../config.py).

### Eval

- To validate the model, you can use `validate.py`. Here is an example to verify the accuracy of pretrained weights.

  ```shell
  python validate.py --model=rexnet_x10 --dataset=imagenet --val_split=val --pretrained
  ```

- To validate the model, you can use `validate.py`. Here is an example to verify the accuracy of your training.

  ```shell
  python validate.py --model=rexnet_x10 --dataset=imagenet --val_split=val --ckpt_path='./rexnetx10_ckpt/rexnet-best.ckpt'
  ```

### Deployment (optional)

Please refer to the deployment tutorial in MindCV.