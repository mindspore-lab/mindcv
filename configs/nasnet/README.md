# NasNet
<!--- Guideline: please use url linked to the paper abstract in ArXiv instead of PDF for fast loading.  -->
> [Learning Transferable Architectures for Scalable Image Recognition](https://arxiv.org/abs/1707.07012)


## Introduction
<!--- Guideline: Introduce the model and architectures. Please cite if you use/adopt paper explanation from others. -->
<!--- Guideline: If an architecture table/figure is available in the paper, please put one here and cite for intuitive illustration. -->

Neural architecture search (NAS) shows the flexibility on model configuration. By doing neural architecture search in a pooling with convolution layer, max pooling and average pooling layer,
the normal cell and the reduction cell are selected to be part of NasNet. Figure 1 shows NasNet architecture for ImageNet, which are stacked with reduction cell and normal cell.
In conclusion, NasNet could achieve better model performance with fewer model parametes and fewer computation cost on image classification
compared with previous state-of-the-art methods on ImageNet-1K dataset.[[1](#references)]

<p align="center">
  <img src="https://user-images.githubusercontent.com/77485245/224208085-0d6e6b91-873d-49cb-ad54-23ea12483d8f.jpeg" width=200 height=400/>
</p>
<p align="center">
  <em>Figure 1. Architecture of Nasnet [<a href="#references">1</a>] </em>
</p>

## Requirements
| mindspore | ascend driver |  firmware   | cann toolkit/kernel |
| :-------: | :-----------: | :---------: | :-----------------: |
|   2.5.0   |   24.1.0      | 7.5.0.3.220 |     8.0.0.beta1     |



## Quick Start

### Preparation

#### Installation
Please refer to the [installation instruction](https://mindspore-lab.github.io/mindcv/installation/) in MindCV.

#### Dataset Preparation
Please download the [ImageNet-1K](https://www.image-net.org/challenges/LSVRC/2012/index.php) dataset for model training and validation.

### Training
<!--- Guideline: Please avoid using shell scripts in the command line. Python scripts preferred. -->

* Distributed Training

It is easy to reproduce the reported results with the pre-defined training recipe. For distributed training on multiple Ascend 910 devices, please run

```shell
# distributed training on multiple NPU devices
msrun --bind_core=True --worker_num 8 python train.py --config configs/nasnet/nasnet_a_4x1056_ascend.yaml --data_dir /path/to/imagenet
```



For detailed illustration of all hyper-parameters, please refer to [config.py](https://github.com/mindspore-lab/mindcv/blob/main/config.py).

**Note:**  As the global batch size  (batch_size x num_devices) is an important hyper-parameter, it is recommended to keep the global batch size unchanged for reproduction or adjust the learning rate linearly to a new global batch size.

* Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on single NPU device
python train.py --config configs/nasnet/nasnet_a_4x1056_ascend.yaml --data_dir /path/to/dataset --distribute False
```

### Validation

To validate the accuracy of the trained model, you can use `validate.py` and parse the checkpoint path with `--ckpt_path`.

```
python validate.py -c configs/nasnet/nasnet_a_4x1056_ascend.yaml --data_dir /path/to/imagenet --ckpt_path /path/to/ckpt
```

## Performance

Our reproduced model performance on ImageNet-1K is reported as follows.

Experiments are tested on Ascend Atlas 800T A2 machines with mindspore 2.5.0 graph mode.


|  model name  |  params(M)   |  cards  |  batch size  |  resolution  |  jit level  |  graph compile  |  ms/step  |   img/s   |  acc@top1  |  acc@top5  |                                               recipe                                               |                                                  weight                                                   |
|:------------:|:------------:|:-------:|:------------:|:------------:|:-----------:|:---------------:|:---------:|:---------:|:----------:|:----------:|:--------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------:|
| nasnet_a_4x1056 | 5.33      | 8     | 256        | 224x224    | O2        | 800s          | 364.55  | 5617.88 | 74.12    | 91.36    | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/nasnet/nasnet_a_4x1056_ascend.yaml) | [weights](https://download-mindspore.osinfra.cn/toolkits/mindcv/nasnet/nasnet_a_4x1056-015ba575c-910v2.ckpt) |



### Notes
- top-1 and top-5: Accuracy reported on the validation set of ImageNet-1K.


## References
<!--- Guideline: Citation format GB/T 7714 is suggested. -->

[1] Zoph B, Vasudevan V, Shlens J, et al. Learning transferable architectures for scalable image recognition[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2018: 8697-8710.
