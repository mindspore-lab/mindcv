# SwinTransformer
***
> [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/pdf/2103.14030.pdf)

## Introduction
***
Challenges in adapting Transformer from language to vision arise from differences
between the two domains, such as large variations in the
scale of visual entities and the high resolution of pixels
in images compared to words in text. To address these
differences, this method propose a hierarchical Transformer whose
representation is computed with Shifted windows.

Swin Transformer has two features:  
1. hierarchical feature structure;  
2. Linear computational complexity to image size.  

The hierarchical feature structure here makes this model applicable to FPN or U-Net models; The linear complexity is due to the use of local window self attention. These characteristics make this model can be used as a general model for various visual tasks. In visual tasks such as object detection and image segmentation, Swin Transformer has obtained SOTA results.
## Benchmark
***

|        |           |           |           |    Pynative     |  Pynative  |     Graph      |   Graph    |           |            |
| :----: | --------- | :-------: | :-------: | :-------------: | :--------: | :------------: | :--------: | :-------: | :--------: |
|        | Model     | Top-1 (%) | Top-5 (%) | train (s/epoch) | Infer (ms) | train(s/epoch) | Infer (ms) | Download  |   Config   |
|  GPU   | swin_tiny | 80.89     | 95.33     |                 |            |                |            | [model]() | [config]() |
| Ascend | swin_tiny | 80.97     | 95.36     |                 |            |                |            |           |            |



## Examples

***

### Train

- The [yaml config files](../../config) that yield competitive results on ImageNet for different models are listed in the `config` folder. To trigger training using preset yaml config. 

  ```shell
  comming soon
  ```


- Here is the example for finetuning a pretrained swin_tiny on CIFAR10 dataset using Adam optimizer.

  ```shell
  python train.py --model=swin_tiny --pretrained --opt=adam --lr=0.001 ataset=cifar10 --num_classes=10 --dataset_download
  ```
  
  Detailed adjustable parameters and their default value can be seen in [config.py](../../config.py)

### Eval

- To validate the model, you can use `validate.py`. Here is an example to verify the accuracy of pretrained weights.

  ```shell
  python validate.py --model=swin_tiny --dataset=imagenet --val_split=val --pretrained
  ```

- To validate the model, you can use `validate.py`. Here is an example to verify the accuracy of your training.

  ```shell
  python validate.py --model=swin_tiny --dataset=imagenet --val_split=val --ckpt_path='./ckpt/swin_tiny-best.ckpt'
  ```

