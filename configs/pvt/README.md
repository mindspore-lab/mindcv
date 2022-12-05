# Pyramid Vision Transformer
> [Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions](https://arxiv.org/abs/2102.12122)

## Introduction
***

PVT is a general backbone network for dense prediction without convolution operation. PVT introduces a pyramid structure in Transformer to generate multi-scale feature maps for dense prediction tasks. PVT uses a gradual reduction strategy to control the size of the feature maps through the patch embedding layer, and proposes a spatial reduction attention (SRA) layer to replace the traditional multi head attention layer in the encoder, which greatly reduces the computing/memory overhead.

![](pvt.png)

## Results
***

| Model           |  Top-1 (%)  | Train T. | Infer T. |  Download | Config | Log |  
|-----------------|-------------|----------|----------|------------|-------|--------|
| PVT_tiny     |74.92 |  433s/epoch  | 16ms/step | [model]() | [cfg]()    | [log]() |
| PVT_small     | 79.66  |538s/epoch |30ms/step | [model]() | [cfg]()    | [log]() |
| PVT_medium    |81.82  |766s/epoch |47ms/step | [model]() | [cfg]()    | [log]() |
| PVT_large    |81.75  |1074s/epoch |67ms/step | [model]() | [cfg]()    | [log]() |

#### Notes
- All models are trained on ImageNet-1K training set and the top-1 accuracy is reported on the validatoin set.
- All models are trained with Ascend910*8 in graph mode and infered with an Ascend310.  

## Quick Start
***
### Preparation

#### Installation
Please refer to the [installation instruction](https://github.com/mindspore-ecosystem/mindcv#installation) in MindCV.
  
#### Dataset Preparation
Please download the [ImageNet-1K](https://www.image-net.org/download.php) dataset for model training and validation.

### Training

- **Hyper-parameters.** The hyper-parameter configurations for producing the reported results are stored in the yaml files in `mindcv/configs/pvt` folder. For example, to train with one of these configurations, you can run:

  ```shell
  # train densenet121 on 8 GPUs
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  mpirun -n 8 python train.py -c configs/pvt/pvt_tiny_ascend.yaml --data_dir /path/to/imagenet
  ```
  
  Note that the number of GPUs/Ascends and batch size will influence the training results. To reproduce the training result at most, it is recommended to use the **same number of GPUs/Ascneds** with the same batch size.

- **Finetuning.** Here is an example for finetuning a pretrained pvt tiny on CIFAR10 dataset using Momentum optimizer.

  ```shell
  python train.py --model=pvt_tiny --pretrained --opt=momentum --lr=0.001 dataset=cifar10 --num_classes=10 --dataset_download
  ```

Detailed adjustable parameters and their default value can be seen in [config.py](../../config.py).

### Validation

- To validate the trained model, you can use `validate.py`. Here is an example for pvt tiny to verify the accuracy of
  pretrained weights.

  ```shell
  python validate.py --model=model=pvt_tiny  --dataset=imagenet --val_split=val --pretrained
  ```

- To validate the model, you can use `validate.py`. Here is an example for pvt tiny  to verify the accuracy of your training.

  ```shell
  python validate.py --model=model=pvt_tiny  --dataset=imagenet --val_split=val --ckpt_path='./ckpt/model=pvt_tiny-best.ckpt'
  ```

### Deployment (optional)

Please refer to the deployment tutorial in MindCV.


  
