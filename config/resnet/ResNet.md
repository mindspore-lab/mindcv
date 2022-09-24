# ResNet
***
> [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)

##Introduction
***
ResNet (residual neural network) was proposed by Kaiming He and other four Chinese of Microsoft Research Institute. Through the use of ResNet unit, it successfully trained 152 layers of neural network, and won the championship in ilsvrc2015. The error rate on top 5 was 3.57%, and the parameter quantity was lower than vggnet, so the effect was very outstanding. Traditional convolution network or full connection network will have more or less information loss. At the same time, it will lead to the disappearance or explosion of gradient, which leads to the failure of deep network training. ResNet solves this problem to a certain extent. By passing the input information to the output, the integrity of the information is protected. The whole network only needs to learn the part of the difference between input and output, which simplifies the learning objectives and difficulties.The structure of ResNet can accelerate the training of neural network very quickly, and the accuracy of the model is also greatly improved. At the same time, ResNet is very popular, even can be directly used in the concept net network.

These are examples of training ResNet18/ResNet50/ResNet101/ResNet152/SE-ResNet50 with CIFAR-10/ImageNet2012 dataset in MindSpore.ResNet50 and ResNet101 can reference [paper 1](https://arxiv.org/pdf/1512.03385.pdf) below, and SE-ResNet50 is a variant of ResNet50 which reference  [paper 2](https://arxiv.org/abs/1709.01507) and [paper 3](https://arxiv.org/abs/1812.01187) below, Training SE-ResNet50 for just 24 epochs using 8 Ascend 910, we can reach top-1 accuracy of 75.9%.(Training ResNet101 with dataset CIFAR-10 and SE-ResNet50 with CIFAR-10 is not supported yet.)

![](figures\resnet网络结构.png)



## Benchmark
***

|              |           |           |      Pynative       |        Pynative        |  Pynative  |        Graph        |         Graph          |   Graph    |           |            |
| :----------: | :-------: | :-------: | :-----------------: | :--------------------: | :--------: | :-----------------: | :--------------------: | :--------: | :-------: | :--------: |
|    Model     | Top-1 (%) | Top-5 (%) | Speed GPU (s/epoch) | Speed Ascend (s/epoch) | Infer (ms) | Speed GPU (s/epoch) | Speed Ascend (s/epoch) | Infer (ms) | Download  |   Config   |
| resnet18 |  78.16     |  94.39  |   0.210    |     0.207           |  40000    |    0.110       |     0.107       |  20000    | [model]() | [config]() |
| resnet34 |  78.47     |  94.40  |   0.108    |     0.111         |  40000    |    0.210       |     0.190       |  20000    | [model]() | [config]() |
| resnet50 |  79.26     |  94.75  |   0.37    |     0.218           |  40000    |    0.27       |     0.118       |  20000    | [model]() | [config]() |
| resnet101 |  80.13     |  95.4  |   0.108    |     0.038        |  40000    |    0.200      |     0.118       |  20000    | [model]() | [config]() |
| resnet152 |  80.62    |  95.51  |   0.120    |     0.047        |  40000    |    0.220      |     0.150       |  20000    | [model]() | [config]() |


## Examples

***

### Train

- The [yaml config files](../../config) that yield competitive results on ImageNet for different models are listed in the `config` folder. To trigger training using preset yaml config. 

  ```shell
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  mpirun --allow-run-as-root -n 8 python train.py -c config/resnet/resnet50.yaml
  
  ```


- Here is the example for finetuning a pretrained Resnet on CIFAR10 dataset using Adam optimizer.

  ```shell
  python train.py --model=resnet50 --pretrained --opt=adam --lr=0.001 ataset=cifar10 --num_classes=10 --dataset_download
  
  ```
  
  Detailed adjustable parameters and their default value can be seen in [config.py](../../config.py)

### Eval

- To validate the model, you can use `validate.py`. Here is an example to verify the accuracy of pretrained weights.

  ```shell
  python validate.py --model=resnet50 --dataset=imagenet --val_split=val --pretrained
  
  ```

- To validate the model, you can use `validate.py`. Here is an example to verify the accuracy of your training.

  ```shell
  python validate.py --model=resnet50 --dataset=imagenet --val_split=val --ckpt_path='./ckpt/resnet50-best.ckpt' 
  ```

