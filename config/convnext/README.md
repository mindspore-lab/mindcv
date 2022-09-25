# ConvNeXt
***
> [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545.pdf)

## Introduction
***
ConvNeXt is a pure ConvNet structure, which imitates the design idea of ViT by standard convolution, through
1. Macro Design;
2. ResNeXt;
3. Inverted Bottleneck;
4. Large Kernel Size;
5. Various Layer wise Micro Design

Finally, the performance of Imagenet dataset classification is superior to SwinTransformer in COCO detection and ADE20K segmentation tasks due to Transformer, while maintaining the simplicity and efficiency of ConvNet.

## Benchmark
***

|        |           |           |           |    Pynative     |  Pynative  |     Graph      |   Graph    |           |            |
| :----: | --------- | :-------: | :-------: | :-------------: | :--------: | :------------: | :--------: | :-------: | :--------: |
|        | Model     | Top-1 (%) | Top-5 (%) | train (s/epoch) | Infer (ms) | train(s/epoch) | Infer (ms) | Download  |   Config   |
|  GPU   | ConvNeXt_tiny | 82.12     | 95.71     |                 |            |                |            | [model]() | [config]() |
| Ascend | ConvNeXt_tiny | 82.20     | 95.74     |                 |            |                |            |           |            |



## Examples

***

### Train

- The [yaml config files](../../config) that yield competitive results on ImageNet for different models are listed in the `config` folder. To trigger training using preset yaml config. 

  ```shell
  comming soon
  ```


- Here is the example for finetuning a pretrained convnext_tiny on CIFAR10 dataset using Adam optimizer.

  ```shell
  python train.py --model=convnext_tiny --pretrained --opt=adam --lr=0.001 ataset=cifar10 --num_classes=10 --dataset_download
  ```
  
  Detailed adjustable parameters and their default value can be seen in [config.py](../../config.py)

### Eval

- To validate the model, you can use `validate.py`. Here is an example to verify the accuracy of pretrained weights.

  ```shell
  python validate.py --model=convnext_tiny --dataset=imagenet --val_split=val --pretrained --interpolation='pilcubic'
  ```

- To validate the model, you can use `validate.py`. Here is an example to verify the accuracy of your training.

  ```shell
  python validate.py --model=convnext_tiny --dataset=imagenet --val_split=val --ckpt_path='./ckpt/convnext_tiny-best.ckpt' --interpolation='pilcubic' 
  ```

