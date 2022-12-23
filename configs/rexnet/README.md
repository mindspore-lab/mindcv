# ReXNet
***
> [ReXNet: Rethinking Channel Dimensions for Efficient Model Design](https://arxiv.org/abs/2007.00992)

##  Introduction
***
This is a new paradigm for network architecture design. ReXNet proposes a set of design principles to solve the Representational Bottleneck problem in the existing network. Rexnet combines these design principles with the existing network units to obtain a new network, RexNet, which achieves a great performance improvement.



## Results
***

| Model           | Context   |  Top-1 (%)  | Top-5 (%)  |  Params (M)    | Train T. | Infer T. |  Download | Config | Log |
|-----------------|-----------|-------|-------|------------|-------|--------|---|--------|--------------|
| rexnet_x09 | D910x8-G | 77.07 | 93.41    |      |   |   | [model](https://download.mindspore.cn/toolkits/mindcv/rexnet/)  | [cfg]() | [log]() |
| rexnet_x10 | D910x8-G | 77.38 | 93.60    |       |   |   | [model](https://download.mindspore.cn/toolkits/mindcv/rexnet/)  | [cfg]() | [log]() |
| rexnet_x13 | D910x8-G | 79.06 | 94.28 |  |   |   | [model](https://download.mindspore.cn/toolkits/mindcv/rexnet/)  | [cfg]() | [log]() |
| rexnet_x15 | D910x8-G | 79.94 | 94.74  |   |   |   | [model](https://download.mindspore.cn/toolkits/mindcv/rexnet/)  | [cfg]() | [log]() |
| rexnet_x20 | D910x8-G | 80.6 | 94.99  |   |   |   | [model](https://download.mindspore.cn/toolkits/mindcv/rexnet/)  | [cfg]() | [log]() |

#### Notes

- All models are trained on ImageNet-1K training set and the top-1 accuracy is reported on the validatoin set.
- Context: GPU_TYPE x pieces - G/F, G - graph mode, F - pynative mode with ms function.  


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
