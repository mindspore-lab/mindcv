# MindSpore Computer Vision

## Introduction
MindSpore Computer Vision is an open source computer vision research toolbox based on MindSpore in computer vision direction. It is mainly used for the development of image tasks and includes a large number of classic and cutting-edge deep learning classification models, such as ResNet, ViT, and SwinTransformer.


### Major Features
- Friendly modular design for the overal DL workflow, including constructing dataloader, models, optimizer, loss for training and testing. It is easy to customize your data transform and learning algorithms. 
- State-of-art models, MindCV provides various SoTA CNN-based and Transformer-based models with pretrained weights including SwinTransformer and EfficientNet (See model list) 
- High efficiency, extensibility and compatibility for different hardware platform  (GPU/CPU/Ascend)

### Results

Under construction... 

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
pip install https://github.com/mindlab-ai/mindcv/releases/download/v0.0.1-alpha/mindcv-0.0.1a0-py3-none-any.whl
```

### Install from source
To install MindCV from source, please run,
```shell
# Clone the mindcv repository.
git clone https://github.com/mindlab-ai/mindcv.git
cd mindcv

# Install
python setup.py install
```

## Get Started 

### Hands-on Demo
Please see the [Quick Start Demo](quick_start.ipynb) to help you get started with MindCV and learn about the basic usage quickly. Below is how to create a deep vision model quickly.  

```python
>>> import mindcv 
# Search a wanted pretrained model 
>>> mindcv.list_models("densenet*", pretrain=True)
['densenet201', 'densenet161', 'densenet169', 'densenet121']
# Create the model object
>>> network = mindcv.create_model('densenet121', pretrained=True)
```

### Quick Running Scripts
It is easy to train your model on standard datasets or your own dataset with MindCV. Model training, transfer learning, or evaluaiton can be done using one or a few line of code with flexible configuration. Below are the running examples for reference.  

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

The [yaml config files](config) that yield competitive results on ImageNet for different models are listed in the `config` folder. To trigger training using preset yaml config, 

```shell
mpirun --allow-run-as-root -n 4 python train.py -c config/squeezenet/squeezenet_1.0_gpu.yaml    
```

- Validation

To validate the model, you can use `validate.py`. Here is an example.
```shell
python validate.py --model=densenet121 --dataset=imagenet --val_split=val \
		           --ckpt_path='./ckpt/densenet121-best.ckpt' 
``` 


## Tutorials
We provide [jupyter notebook tutorials](tutorials) for  

- [Learn about configs](tutorials/learn_about_config.ipynb)  
- [Inference with a pretrained model](tutorials/inference.ipynb) 
- [Finetune a pretrained model on custom datasets](tutorials/finetune.ipynb) 
- [Customize models](tutorials/customize_model.ipynb) //tbc
- [Optimizing performance for vision transformer](tutorials/transformer.ipynb) //tbc


## Notes
### What is New 

- 2022/09/13
1. Add Adan optimizer (experimental), tested in non-dist graph mode. 

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
