# Res2Net
***
> [Res2Net: A New Multi-scale Backbone Architecture](https://arxiv.org/pdf/1904.01169.pdf)

##  Introduction
***
GoogLeNet is a new deep learning structure proposed by Christian Szegedy in 2014. Prior to this, AlexNet, VGG and other structures achieved better training effects by increasing the depth (number of layers) of the network, but the increase in the number of layers It will bring many negative effects, such as overfit, gradient disappearance, gradient explosion, etc. The proposal of inception improves the training results from another perspective: it can use computing resources more efficiently, and can extract more features under the same amount of computing, thereby improving the training results.

![](res2net.png)



## Benchmark
***

|        |                |           |           |    Pynative     |  Pynative  |     Graph      |   Graph    |           |            |
| :----: | -------------- | :-------: | :-------: | :-------------: | :--------: | :------------: | :--------: | :-------: | :--------: |
|        | Model          | Top-1 (%) | Top-5 (%) | train (s/epoch) | Infer (ms) | train(s/epoch) | Infer (ms) | Download  |   Config   |
|  GPU   | res2net50      |           |           |                 |            |                |            | [model]() | [config]() |
| Ascend | res2net50      |           |           |                 |            |                |            |           |            |
|  GPU   | res2net101     |           |           |                 |            |                |            | [model]() | [config]() |
| Ascend | res2net101     |           |           |                 |            |                |            |           |            |
|  GPU   | res2net50_v1b  |           |           |                 |            |                |            | [model]() | [config]() |
| Ascend | res2net50_v1b  |           |           |                 |            |                |            |           |            |
|  GPU   | res2net101_v1b |           |           |                 |            |                |            | [model]() | [config]() |
| Ascend | res2net101_v1b |           |           |                 |            |                |            |           |            |



## Examples

***

### Train

- The [yaml config files](../../configs) that yield competitive results on ImageNet for different models are listed in the `configs` folder. To trigger training using preset yaml config. 

  ```shell
  comming soon
  ```


- Here is the example for finetuning a pretrained InceptionV3 on CIFAR10 dataset using Adam optimizer.

  ```shell
  python train.py --model=res2net50 --pretrained --opt=momentum --lr=0.001 dataset=cifar10 --num_classes=10 --dataset_download
  ```

Detailed adjustable parameters and their default value can be seen in [config.py](../../config.py).

### Eval

- To validate the model, you can use `validate.py`. Here is an example for res2net50 to verify the accuracy of pretrained weights.

  ```shell
  python validate.py --model=res2net50 --dataset=imagenet --val_split=val --pretrained
  ```

- To validate the model, you can use `validate.py`. Here is an example for res2net50 to verify the accuracy of your training.

  ```shell
  python validate.py --model=res2net50 --dataset=imagenet --val_split=val --ckpt_path='./ckpt/res2net50-best.ckpt'
  ```

