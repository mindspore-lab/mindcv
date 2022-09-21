# MindSpore Computer Vision

MindSpore Computer Vision is an open source computer vision research toolbox based on MindSpore in computer vision direction. It is mainly used for the development of image tasks and includes a large number of classic and cutting-edge deep learning classification models, such as ResNet, ViT, and SwinTransformer.

## Major Features

- Various backbones based on CNN and Transformer and pretrained models
- High efficiency, extensibility and compatibility(GPU/CPU/Ascend)
- Easy-to-use API

## Base Structure

MindSpore Computer Vision, a MindSpore base Python package, provides high-level features:

- Base backbone of models like resnet and mobilenet series.
- Deep neural networks' workflows which contain loss, optimizers, lr_scheduler.
- Domain oriented rich dataset interface.

## Dependency

- mindspore >= 1.8.1
- numpy >= 1.17.0
- pyyaml >= 5.3
- tqdm
- openmpi 4.0.3 (for distributed mode) 

## Installation

The following instructions assume that you have desired dependency installed and working. 

```shell
pip install https://github.com/mindlab-ai/mindcv/releases/download/v0.0.1-alpha/mindcv-0.0.1a0-py3-none-any.whl
```

- From source:

```shell
# Clone the mindcv repository.
git clone https://github.com/mindlab-ai/mindcv.git
cd mindcv

# Install
python setup.py install
```

## Get Started
See [Get Started With MindCV](quick_tour.ipynb)  to learn about basic usage.


## Changes 

### 2022/09/13
* optim factory 
1. Add Adan optimizer (experimental), tested in non-dist graph mode. 

* dataset factory 
1. Adjust interface for custom data. 

* test
1. unit test for optim and models
2. test scripts for finetuning and customized dataset 

### 2022/09/06

* loss factory 
1. Remove args param, use detailed loss-related params
2. Add BCE loss
3. add class weighted loss support, weighted BCE loss and weighted CE loss. 
4. change arg name `smooth_factor` to `label_smoothing`  
5. change loss type and param organization structure. 
6. add test code 

* optim factory 
1. adjust APIs, remove args
2. add adam, adamW, lamb, adagrad

* scheduler factory
1. reorganize code, remove args
2. support Step decay LR, exponential decay LR. 

* TODOs:
1. test loss computation correctness 
2. check adamW difference compared to pytorch
3. label smoothing support for BCE
4. label data type changed to float for using BCE loss



## License

This project is released under the [Eclipse Public License 1.0](LICENSE).

## Feedbacks and Contact

The dynamic version is still under development, if you find any issue or have an idea on new features, please don't hesitate to contact us via [issue](https://github.com/mindlab-ai/mindcv/issues).

## Acknowledgement

MindSpore is an open source project that welcome any contribution and feedback. We wish that the toolbox and benchmark could serve the growing research community by providing a flexible as well as standardized toolkit to reimplement existing methods and develop their own new computer vision methods.

## Contributing

We appreciate all contributions to improve MindSpore Vision. Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for the contributing guideline.

## Citation

If you find this project useful in your research, please consider citing:

```latex
@misc{MindSpore Computer Vision 2022,
    title={{MindSpore Computer  Vision}:MindSpore Computer Vision Toolbox and Benchmark},
    author={MindSpore Vision Contributors},
    howpublished = {\url{https://github.com/mindlab-ai/mindcv/}},
    year={2022}
}
```
