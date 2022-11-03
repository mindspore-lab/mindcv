# DenseNet
> [Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993.pdf)

## Introduction
***

Recent work has shown that convolutional networks can be substantially deeper, more accurate, and efficient to train if
they contain shorter connections between layers close to the input and those close to the output. Dense Convolutional
Network (DenseNet) is introduced based on this observation, which connects each layer to every other layer in a
feed-forward fashion. Whereas traditional convolutional networks with $L$ layers have $L$ connections-one between each
layer and its subsequent layer, our network has $\frac{L(L+1)}{2}$ direct connections. For each layer, the feature-maps
of all preceding layers are used as inputs, and its own feature-maps are used as inputs into all subsequent layers.
DenseNets have several compelling advantages: they alleviate the vanishing-gradient problem, strengthen feature
propagation, encourage feature reuse, and substantially reduce the number of parameters.

![](densenet.png)

## Results
***

| Model           | Context   |  Top-1 (%)  | Top-5 (%)  |  Params (M)    | Train T. | Infer T. |  Download | Config | Log |  
|-----------------|-----------|-------|-------|------------|-------|--------|---|--------|--------------|
| DenseNet121     |  D910x8-G  | 75.60 | 92.73 | 7.05 |  2min/epoch  | 20ms/step | [model]() | [cfg]()    | [log]() |
| DenseNet169     |  V100x8-F  |       |      | 3.54 | | | [model]() | [cfg]()    | [log]() |

#### Notes
- All models are trained on ImageNet-1K training set and the top-1 accuracy is reported on the validatoin set.
- Context: GPU_TYPE x pieces - G/F, G - graph mode, F - pynative mode with ms function.  

## Quick Start
***
### Preparation

#### Dependency
  ```shell
  pip install mindcv
  ```
  
#### Dataset
```shell
  wget imagenet1k.zip
  ```
### Train

- The [yaml config files](../../configs) that yield competitive results on ImageNet for different models are listed in
  the `configs` folder. To trigger training using preset yaml config.

  ```shell
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  mpirun -n 8 python train.py -c configs/densenet/densenet_121_gpu.yaml --data_dir /path/to/imagenet
  ```

- Here is the example for finetuning a pretrained densenet121 on CIFAR10 dataset using Momentum optimizer.

  ```shell
  python train.py --model=densenet121 --pretrained --opt=momentum --lr=0.001 dataset=cifar10 --num_classes=10 --dataset_download
  ```

Detailed adjustable parameters and their default value can be seen in [config.py](../../config.py).

### Eval

- To validate the model, you can use `validate.py`. Here is an example for densenet121 to verify the accuracy of
  pretrained weights.

  ```shell
  python validate.py --model=densenet121 --dataset=imagenet --val_split=val --pretrained
  ```

- To validate the model, you can use `validate.py`. Here is an example for densenet121 to verify the accuracy of your
  training.

  ```shell
  python validate.py --model=densenet121 --dataset=imagenet --val_split=val --ckpt_path='./ckpt/densenet121-best.ckpt'
  ```

### Deploy (optional)

  ```shell
  python ms_serving.py --model=densenet121 --ckpt_path='./ckpt/densenet121-best.ckpt'
  ```


  
