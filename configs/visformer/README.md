# Visformer
> [Visformer: The Vision-friendly Transformer](https://arxiv.org/pdf/2104.12533.pdf)

## Introduction
***

The past few years have witnessed the rapid development of applying the Transformer module to vision problems. While some
researchers have demonstrated that Transformer based models enjoy a favorable ability of fitting data, there are still 
growing number of evidences showing that these models suffer over-fitting especially when the training data is limited. 
This paper offers an empirical study by performing step-bystep operations to gradually transit a Transformer-based model
to a convolution-based model. The results we obtain during the transition process deliver useful messages for improving 
visual recognition. Based on these observations, we propose a new architecture named Visformer, which is abbreviated from
the ‘Vision-friendly Transformer’.

![](visformer.png)

## Results
***

| Model            | Context   |  Top-1 (%)  | Top-5 (%)  |  Params (M)    | Train T.   | Infer T. |  Download | Config | Log |
|------------------|-----------|-------------|------------|----------------|------------|----------|-----------|--------|-----|
| visformer_tiny   | D910x8-G  | 78.28       | 94.15      | 10             | 496/epoch  | 300.7ms/step | [model](https://download.mindspore.cn/toolkits/mindcv/visformer/) | [cfg]() | [log]() |
| visformer_tiny2  | D910x8-G  | 78.82       | 94.41      | 9              | 390s/epoch | 602.5ms/step | [model](https://download.mindspore.cn/toolkits/mindcv/visformer/) | [cfg]() | [log]() |
| visformer_small  | D910x8-G  | 81.73       | 95.88      | 40             | 445s/epoch | 155.9ms/step | [model](https://download.mindspore.cn/toolkits/mindcv/visformer/) | [cfg]() | [log]() |
| visformer_small2 | D910x8-G  | 82.17       | 95.90      | 23             | 440s/epoch | 153.1ms/step | [model](https://download.mindspore.cn/toolkits/mindcv/visformer/) | [cfg]() | [log]() |

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

- **Hyper-parameters.** The hyper-parameter configurations for producing the reported results are stored in the yaml 
  files in `mindcv/configs/visformer` folder. For example, to train with one of these configurations, you can run:

  ```shell
  # train densenet121 on 8 GPUs
  mpirun -n 8 python train.py --config configs/visformer/visformer_tiny_ascend.yaml --data_dir /path/to/imagenet
  ```

  Note that the number of GPUs/Ascends and batch size will influence the training results. To reproduce the training result at most, it is recommended to use the **same number of GPUs/Ascends** with the same batch size.

Detailed adjustable parameters and their default value can be seen in [config.py](../../config.py).

### Validation

- To validate the model, you can use `validate.py`. Here is an example for visformer_tiny to verify the accuracy of your
  training.

  ```shell
  python validate.py --config configs/visformer/visformer_tiny_ascend.yaml --data_dir /path/to/imagenet --ckpt_path /path/to/visformer_tiny.ckpt
  ```

### Deployment (optional)

Please refer to the deployment tutorial in MindCV.



