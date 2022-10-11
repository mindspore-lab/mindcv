# ReXNet
***
> [ReXNet: Rethinking Channel Dimensions for Efficient Model Design](https://arxiv.org/abs/2007.00992)

##  Introduction
***
Rank Expansion Networks (ReXNets) follow a set of new design principles for designing bottlenecks in image classification models. Authors refine each layer by 1) expanding the input channel size of the convolution layer and 2) replacing the ReLU6s.

## Benchmark
***

|        |           |           |           |    Pynative     |  Pynative  |     Graph      |   Graph    |           |            |
| :----: | --------- | :-------: | :-------: | :-------------: | :--------: | :------------: | :--------: | :-------: | :--------: |
|        | Model     | Top-1 (%) | Top-5 (%) | train (s/epoch) | Infer (ms) | train(s/epoch) | Infer (ms) | Download  |   Config   |
|  GPU   | rexnet_x09 |           |           |          |            |         |            | [model]() | [config]() |
| Ascend | rexnet_x09 |           |           |          |            |         |            |           |            |
|  GPU   | rexnet_x10 |           |           |          |            |         |            | [model]() | [config]() |
| Ascend | rexnet_x10 |           |           |          |            |         |            |           |            |
|  GPU   | rexnet_x13 |           |           |          |            |         |            | [model]() | [config]() |
| Ascend | rexnet_x13 |           |           |          |            |         |            |           |            |
|  GPU   | rexnet_x15 |           |           |          |            |         |            | [model]() | [config]() |
| Ascend | rexnet_x15 |           |           |          |            |         |            |           |            |
|  GPU   | rexnet_x20 |           |           |          |            |         |            | [model]() | [config]() |
| Ascend | rexnet_x20 |           |           |          |            |         |            |           |            |



## Examples

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
