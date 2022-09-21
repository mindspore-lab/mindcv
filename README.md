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


The following instructions assume that you have desired dependency installed and working. 

### Install with pip

```shell
pip install https://github.com/mindlab-ai/mindcv/releases/download/v0.0.1-alpha/mindcv-0.0.1a0-py3-none-any.whl
```

### Install from source

```shell
# Clone the mindcv repository.
git clone https://github.com/mindlab-ai/mindcv.git
cd mindcv

# Install
python setup.py install
```

## Get Started 

### Demo
You can see the notebook demo ([Get Started With MindCV](quick_tour.ipynb)) to learn about the basic usage of MindCV . 


### Quick Running Scripts
It is easy to train your model on standard datasets or your own dataset with MindCV. 

- Standalone Training

You can run `train.py` to do training with customized hyper-parameters. Here is the example for training a DenseNet on CIFAR10 dataset.
``` shell
python train.py --model=densenet121 --optimizer=adam --lr=0.001 \
		--dataset=cifar10 --num_classes=10 --dataset_download    
```

Detailed adjustable hyper-parameters for data transform, model, loss, and optimizer configuration can be viewed in [config.py](config.py)

- Validation

To validate, you can run `validate.py` as shown in the following example.
```shell
python validate.py --model=densenet121 --dataset=cifar10 --val_split=test \
		   --num_classes=10 --dataset_download
``` 

- Distributed Training

For large datasets like ImageNet, it is necessary to do training in distributed mode on multiple devices, which is well supported in MindCV. The following script is an example for training DenseNet121 on ImageNet with 4 GPUs.   

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3  # suppose there are 4 GPUs
mpirun --allow-run-as-root -n 4 python train.py --distribute \
	--model=densenet121 --dataset=imagenet --data_dir=./datasets/imagenet   
```

- Train with Yaml Config

We also provide that yaml config files that yield competitive results on ImageNet for different models in [config](configs) folder. To trigger training using yaml config, 

```shell
mpirun --allow-run-as-root -n 4 python train.py -c configs/squeezenet/squeezenet_1.0_gpu.yaml    
```


## Tutorials
We provide [jupyter notebook tutorials](tutorials) for  

- [Learn about configs](tutorials/learn_about_config.ipynb)  //tbc
- [Inference with a pretrained model](tutorials/inference.ipynb) //tbc
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

MindSpore is an open source project that welcome any contribution and feedback. We wish that the toolbox and benchmark could serve the growing research community by providing a flexible as well as standardized toolkit to reimplement existing methods and develop their own new computer vision methods.

### Contributing

We appreciate all contributions to improve MindSpore Vision. Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for the contributing guideline.

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
