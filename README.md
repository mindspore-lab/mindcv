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

Transfer learning or training can be done easily with the provided scripts.

```
# transfer learning in one command line
python train.py --model=swin_tiny --pretrained --opt=adamw --lr=0.001 --data_dir=data/my_dataset 
```

- **State-of-art models and methods.** MindCV provides various CNN-based and Transformer-based vision models including SwinTransformer. Their pretrained weights and performance reports are provided to help users select and reuse the right model: 

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

To get started with MindCV, please see the [transfer learning tutorial](tutorials/finetune.ipynb), which will give a quick tour on each key component and the train/validate/predict pipelines in MindCV. 

Below is a few code snippets for your taste. 

```python
>>> import mindcv 
# List and find a pretrained vision model 
>>> mindcv.list_models("swin*", pretrained=True)
['swin_tiny']
# Create the model object
>>> network = mindcv.create_model('swin_tiny', pretrained=True)
# Validate its accuracy
>>> !python validate.py --model=swin_tiny --pretrained --dataset=imagenet --val_split=validation
{'Top_1_Accuracy': 0.808343989769821, 'Top_5_Accuracy': 0.9527253836317136, 'loss': 0.8474242982580839}
```

**Image classification demo**

<p align="left">
  <img src="./tutorials/dog.jpg" width=360 />
</p>

```python
>>> !python infer.py --model=swin_tiny --image_path='./tutorials/dog.jpg'
{'Labrador retriever': 0.5700152, 'golden retriever': 0.034551315, 'kelpie': 0.010108651, 'Chesapeake Bay retriever': 0.008229004, 'Walker hound, Walker foxhound': 0.007791956}
```
The top-1 prediction is labrador retriever (拉布拉多犬), which is the right breed of the cut dog.

### Useful Scripts
It is easy to train your model on standard datasets or your own dataset with MindCV. Model training, transfer learning, or evaluaiton can be done using one or a few line of code with flexible configuration. 

- Standalone Training

`train.py` is the main script for model training, where you can configure each component of the training pipline easily. Here is the example for finetuning a pretrained DenseNet on CIFAR10 dataset using Adam optimizer.
``` shell
python train.py --model=resnet50 --pretrained --opt=adam --lr=0.0001 \
		--dataset=cifar10 --dataset_download  --epoch_size=10  
```

**Validation while training.** To track the validation accuracy change during traing, please enable `--val_while_train`, for example

```python
python train.py --model=resnet50 --pretrained --dataset=cifar10 \
		--val_while_train --val_split=test --val_interval=1
``` 

The training loss and validation accuracy for each epoch  will be saved in `{ckpt_save_dir}/results.log`.

Detailed adjustable parameters and their default value can be seen in [config.py](config.py)

- Distributed Training

For large datasets like ImageNet, it is necessary to do training in distributed mode on multiple devices, which is well supported in MindCV. The following script is an example for training DenseNet121 on ImageNet with 4 GPUs.   

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3  # suppose there are 4 GPUs
mpirun --allow-run-as-root -n 4 python train.py --distribute \
	--model=densenet121 --dataset=imagenet --data_dir=./datasets/imagenet   
```

- Configuration

You can either configure your model or other components by specifying external parameters or a yaml config file, for example:

```shell
mpirun --allow-run-as-root -n 4 python train.py -c configs/squeezenet/squeezenet_1.0_gpu.yaml    
```

Reference [yaml config files](configs) that yield competitive results on ImageNet are in the `configs` folder. 


- Validation

It is easy to validate a trained model with `validate.py`. 
```python
# validate a trained checkpoint
python validate.py --model=resnet50 --dataset=imagenet --val_split=validation \
		           --ckpt_path='./ckpt/densenet121-best.ckpt' 

# validate a pretrained SwinTransformer model 
python validate.py --model=swin_tiny --dataset=imagenet --val_split=validation \
		           --pretrained
``` 

For more examples, see [examples/scripts](examples/scripts). 

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
