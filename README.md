# MindCV

<p align="left">
    <a href="https://mindcv-ai.readthedocs.io/en/latest">
        <img alt="docs" src="https://img.shields.io/badge/docs-latest-blue">
    </a>
    <a href="https://github.com/IDEA-Research/detrex/blob/main/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/mindspore-ecosystem/mindcv.svg?color=blue">
    </a>
    <a href="https://github.com/mindspore-ecosystem/mindcv/pulls">
        <img alt="PRs Welcome" src="https://img.shields.io/badge/PRs-welcome-pink.svg">
    </a>
    <a href="https://github.com/mindspore-ecosystem/mindcv/issues">
        <img alt="open issues" src="https://img.shields.io/github/issues/mindspore-ecosystem/mindcv">
    </a>
    <a href="https://github.com/mindspore-ecosystem/mindcv/tags">
        <img alt="GitHub tags" src="https://img.shields.io/github/tag/mindspore-ecosystem/mindcv.svg">
    </a>
</p>

[Introduction](#introduction) |
[Installation](#installation) |
[Get Started](#get-started) |
[Tutorials](#tutorials) |
[Notes](#notes) 

## Introduction
MindCV is an open source toolbox for computer vision research and development based on [MindSpore](https://www.mindspore.cn/en). It collects a series of classic and SoTA vision models, such as ResNet and SwinTransformer, along with their pretrained weights. SoTA methods like MixUp, AutoAugment are also provided for performance improvement. With the decoupled module design, it is easy to apply or adapt MindCV to your own CV tasks. 

<details open>
<summary> Major Features </summary>
	
- **Easy-to-use.** MindCV decomposes the vision framework into multiple components, each of which can be configured in one line of code. It is easy to customize your data pipeline, models, and learning pipeline with MindCV: 

```python
>>> import mindcv 
>>> network = mindcv.create_model('resnet50', pretrained=True)
```

MindCV also provides strong train and validatio engines, which allow users custmize each component easily and complete their training or transfer learning task in one line of script.

```
# transfer learning in one command line
python train.py --model=swin_tiny --pretrained --opt=adamw --lr=0.001 --dataset=cifar10 --dataset_download 
```

- **State-of-art models and methods.** MindCV provides various CNN-based and Transformer-based vision models including SwinTransformer. Their pretrained weights and performance reports are provided to help users select and reuse the right model: 
```
# validate the pretrained swin transformer (tiny)
python validate.py --model=swin_tiny --pretrained --dataset=imagenet --val_split=validation
>>> {'Top_1_Accuracy': 0.808343989769821, 'Top_5_Accuracy': 0.9527253836317136, 'loss': 0.8474242982580839}
``` 

- **Flexibility and efficiency.** MindCV is bulit on MindSpore which is an efficent DL framework that can run on different hardward platforms (GPU/CPU/Ascend). It supports both graph mode for high efficiency and pynative mode for flexibity.
	
</details>
	
### Benchmark Results

Coming soon.


## Installation

### Dependency

- mindspore >= 1.8.1
- numpy >= 1.17.0
- pyyaml >= 5.3
- tqdm
- openmpi 4.0.3 (for distributed mode) 

To install the dependency, please run
```shell
pip install -r requirements.txt
```

MindSpore can be easily installed by following the official [instruction](https://www.mindspore.cn/install) where you can select your hardware platform for the best fit. To run in distributed mode, [openmpi](https://www.open-mpi.org/software/ompi/v4.0/) is required to install.   

The following instructions assume the desired dependency is fulfilled. 

### Install with pip
MindCV can be installed with pip. 
```shell
pip install https://github.com/mindlab-ai/mindcv/releases/download/v0.0.1-beta/mindcv-0.0.1b0-py3-none-any.whl
```

### Install from source
To install MindCV from source, please run,
```shell
pip install git+https://github.com/mindlab-ai/mindcv.git
```

## Get Started 

### Hands-on Tutorial
Please see the [Quick Start Demo](quick_start.ipynb) to help you get started with MindCV and learn about the basic usage quickly. 

You can also see the [finetune tutorial](tutorials/finetune.ipynb) to learn how to apply a pretrained SoTA model to your own classification task. 

Below is how to find and  create a deep vision model quickly.  

```python
>>> import mindcv 
# Search a wanted pretrained model 
>>> mindcv.list_models("densenet*", pretrained=True)
['densenet201', 'densenet161', 'densenet169', 'densenet121']
# Create the model object
>>> network = mindcv.create_model('densenet121', pretrained=True)
```

### Useful Scripts
It is easy to train your model on standard datasets or your own dataset with MindCV. Model training, transfer learning, or evaluaiton can be done using one or a few line of code with flexible configuration. 

- Standalone Training

`train.py` is the main script for model training, where you can set the dataset, data transformation, model, loss, LR scheduler, and optimizier easily. Here is the example for finetuning a pretrained DenseNet on CIFAR10 dataset using Adam optimizer.
``` shell
python train.py --model=densenet121 --pretrained --opt=adam --lr=0.001 \
		--dataset=cifar10 --num_classes=10 --dataset_download    
```

Detailed adjustable parameters and their default value can be seen in [config.py](config.py)

- Distributed Training

For large datasets like ImageNet, it is necessary to do training in distributed mode on multiple devices, which is well supported in MindCV. The following script is an example for training DenseNet121 on ImageNet with 4 GPUs.   

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3  # suppose there are 4 GPUs
mpirun --allow-run-as-root -n 4 python train.py --distribute \
	--model=densenet121 --dataset=imagenet --data_dir=./datasets/imagenet   
```

- Train with Yaml Config

The [yaml config files](configs) that yield competitive results on ImageNet for different models are listed in the `configs` folder. To trigger training using preset yaml config, 

```shell
mpirun --allow-run-as-root -n 4 python train.py -c configs/squeezenet/squeezenet_1.0_gpu.yaml    
```

- Validation

To validate a trained/pretrained model, you can use `validate.py`. 
```python
# validate a trained checkpoint
python validate.py --model=resnet50 --dataset=imagenet --val_split=validation \
		           --ckpt_path='./ckpt/densenet121-best.ckpt' 

# validate a pretrained SwinTransformer model 
python validate.py --model=swin_tiny --dataset=imagenet --val_split=validation \
		           --pretrained
>>> {'Top_1_Accuracy': 0.808343989769821, 'Top_5_Accuracy': 0.9527253836317136, 'loss': 0.8474242982580839}
``` 

- Validate during Training

Validation during training can be enabled by setting the `--val_while_train` argument, e.g.,
```python
python train.py --model=resnet18 --dataset=cifar10 --val_while_train --val_split=test --val_interval=1
``` 
The training loss and validation accuracy for each epoch  will be saved in `./CKPT_NAME/metric_log.txt`.


You can use `mindcv.list_models()` to find out all supported models. It is easy to apply any of them to your  tasks with these scripts. For more examples, see [examples/scripts](examples/scripts). 

## Tutorials
We provide [jupyter notebook tutorials](tutorials) for  

- [Learn about configs](tutorials/learn_about_config.ipynb)  
- [Inference with a pretrained model](tutorials/inference.ipynb) 
- [Finetune a pretrained model on custom datasets](tutorials/finetune.ipynb) 
- [Customize models] //coming soon
- [Optimizing performance for vision transformer] //coming soon
- [Deployment demo](tutorials/deployment.ipynb) 


## Notes
### What is New 
- 2022/10/12
1. Both BCE and CE loss now support class-weight config, label smoothing, and auxilary logit input (for networks like inception).
- 2022/09/13
1. Add Adan optimizer (experimental) 

### License

This project is released under the [Apache License 2.0](LICENSE.md).

### Feedbacks and Contact

The dynamic version is still under development, if you find any issue or have an idea on new features, please don't hesitate to contact us via [issue](https://github.com/mindlab-ai/mindcv/issues).

### Acknowledgement

MindCV is an open source project that welcome any contribution and feedback. We wish that the toolbox and benchmark could serve the growing research community by providing a flexible as well as standardized toolkit to reimplement existing methods and develop their own new computer vision methods.

### Contributing

We appreciate all contributions to improve MindCV. Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for the contributing guideline.

### Citation

If you find this project useful in your research, please consider citing:

```latex
@misc{MindSpore Computer Vision 2022,
    title={{MindSpore Computer  Vision}:MindSpore Computer Vision Toolbox and Benchmark},
    author={MindSpore Vision Contributors},
    howpublished = {\url{https://github.com/mindlab-ai/mindcv/}},
    year={2022}
}
```
