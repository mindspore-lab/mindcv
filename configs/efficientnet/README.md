# EfficientNet
<!--- Guideline: please use url linked to the paper abstract in ArXiv instead of PDF for fast loading.  -->
> [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)

## Introduction
<!--- Guideline: Introduce the model and architectures. Please cite if you use/adopt paper explanation from others. -->
<!--- Guideline: If an architecture table/figure is available in the paper, please put one here and cite for intuitive illustration. -->

Figure 1 shows the methods from three dimensions -- width, depth, resolution and compound to expand the model. Increasing the model
size solely would cause the model performance to sub-optimal solution. Howerver, if three methods could be applied together into the model
, it is more likely to achieve optimal solution. By using neural architecture search, the best configurations for width scaling, depth scaling
and resolution scaling could be found. EfficientNet could achieve better model performance on ImageNet-1K dataset compared with previous methods.[[1](#references)]

<p align="center">
  <img src="https://user-images.githubusercontent.com/77485245/225044036-d0344404-e86c-483c-971f-863ebe6decc6.jpeg" width=800 />
</p>
<p align="center">
  <em>Figure 1. Architecture of Efficientent [<a href="#references">1</a>] </em>
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
msrun --bind_core=True --worker_num 8 python train.py --config configs/efficientnet/efficientnet_b0_ascend.yaml --data_dir /path/to/imagenet
```



For detailed illustration of all hyper-parameters, please refer to [config.py](https://github.com/mindspore-lab/mindcv/blob/main/config.py).

**Note:**  As the global batch size  (batch_size x num_devices) is an important hyper-parameter, it is recommended to keep the global batch size unchanged for reproduction or adjust the learning rate linearly to a new global batch size.

* Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on single NPU device
python train.py --config configs/efficientnet/efficientnet_b0_ascend.yaml --data_dir /path/to/dataset --distribute False
```

### Validation

To validate the accuracy of the trained model, you can use `validate.py` and parse the checkpoint path with `--ckpt_path`.

```
python validate.py -c configs/efficientnet/efficientnet_b0_ascend.yaml --data_dir /path/to/imagenet --ckpt_path /path/to/ckpt
```

## Performance

Our reproduced model performance on ImageNet-1K is reported as follows.

Experiments are tested on ascend 910* with mindspore 2.5.0 graph mode.

| model name      | params(M) | cards | batch size | resolution | jit level | graph compile | ms/step | img/s   | acc@top1 | acc@top5 | recipe                                                                                                     | weight                                                                                                            |
| --------------- | --------- | ----- | ---------- | ---------- | --------- | ------------- |---------| ------- | -------- | -------- | ---------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| efficientnet_b0 | 5.33      | 8     | 128        | 224x224    | O2        | 353s          | 173.47  | 5903.03| 76.88    | 93.28    | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/efficientnet/efficientnet_b0_ascend.yaml) | [weights](https://download-mindspore.osinfra.cn/toolkits/mindcv/efficientnet/efficientnet_b0-f8d7aa2a-910v2.ckpt) |

### Notes
- top-1 and top-5: Accuracy reported on the validation set of ImageNet-1K.


## References
<!--- Guideline: Citation format GB/T 7714 is suggested. -->

[1] Tan M, Le Q. Efficientnet: Rethinking model scaling for convolutional neural networks[C]//International conference on machine learning. PMLR, 2019: 6105-6114.
