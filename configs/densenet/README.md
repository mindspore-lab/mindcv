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
| DenseNet121 | D910x8-G | 75.64     | 92.84     | 8.06       | 238s/epoch | 6.7ms/step | [model]() | [cfg]() | [log]() |
| DenseNet161 | D910x8-G | 79.09     | 94.66     | 28.90      | 472s/epoch | 8.7ms/step | [model]() | [cfg]() | [log]() |
| DenseNet169 | D910x8-G | 77.26 | 93.71 | 14.30 | 313s/epoch | 7.4ms/step | [model]() | [cfg]() | [log]() |
| DenseNet201 | D910x8-G | 78.14 | 94.08 | 20.24 | 394s/epoch | 7.9ms/step | [model]() | [cfg]() | [log]() |

#### Notes

- All models are trained on ImageNet-1K training set and the top-1 accuracy is reported on the validatoin set.
- Context: GPU_TYPE x pieces - G/F, G - graph mode, F - pynative mode with ms function.  

## Quick Start
***
### Preparation

#### Installation
Please refer to the [installation instruction](https://github.com/mindspore-ecosystem/mindcv#installation) in MindCV.

#### Dataset Preparation
Please download the [ImageNet-1K](https://www.image-net.org/download.php) dataset for model training and validation.

### Training

- **Hyper-parameters.** The hyper-parameter configurations for producing the reported results are stored in the yaml files in `mindcv/configs/densenet` folder. For example, to train with one of these configurations, you can run:

  ```shell
  # train densenet121 on 8 GPUs
  mpirun -n 8 python train.py --config configs/densenet/densenet_121_gpu.yaml --data_dir /path/to/imagenet
  ```

  Note that the number of GPUs/Ascends and batch size will influence the training results. To reproduce the training result at most, it is recommended to use the **same number of GPUs/Ascends** with the same batch size.

Detailed adjustable parameters and their default value can be seen in [config.py](../../config.py).

### Validation

- To validate the model, you can use `validate.py`. Here is an example for densenet121 to verify the accuracy of your
  training.

  ```shell
  python validate.py --config configs/densenet/densenet_121_gpu.yaml --data_dir /path/to/imagenet --ckpt_path /path/to/densenet121.ckpt
  ```

### Deployment (optional)

Please refer to the deployment tutorial in MindCV.



