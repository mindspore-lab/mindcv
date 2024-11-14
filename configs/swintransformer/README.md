# Swin Transformer

<!--- Guideline: please use url linked to the paper abstract in ArXiv instead of PDF for fast loading.  -->
> [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)



## Introduction

<!--- Guideline: Introduce the model and architectures. Please cite if you use/adopt paper explanation from others. -->
<!--- Guideline: If an architecture table/figure is available in the paper, please put one here and cite for intuitive illustration. -->

The key idea of Swin transformer is that the features in shifted window go through transformer module rather than the
whole feature map.
Besides that, Swin transformer extracts features of different levels. Additionally, compared with Vision Transformer (
ViT), the resolution
of Swin Transformer in different stages varies so that features with different sizes could be learned. Figure 1 shows
the model architecture
of Swin transformer. Swin transformer could achieve better model performance with smaller model parameters and less
computation cost
on ImageNet-1K dataset compared with ViT and ResNet.[[1](#references)]

<p align="center">
  <img src="https://user-images.githubusercontent.com/77485245/225558829-4cbdc830-5141-41f4-836b-8f1bc771ca2f.png" />
</p>
<p align="center">
  <em>Figure 1. Architecture of Swin Transformer [<a href="#references">1</a>] </em>
</p>

## Requirements
| mindspore | ascend driver |  firmware   | cann toolkit/kernel |
| :-------: | :-----------: | :---------: | :-----------------: |
|   2.3.1   |   24.1.RC2    | 7.3.0.1.231 |    8.0.RC2.beta1    |



## Quick Start

### Preparation

#### Installation

Please refer to the [installation instruction](https://mindspore-lab.github.io/mindcv/installation/) in MindCV.

#### Dataset Preparation

Please download the [ImageNet-1K](https://www.image-net.org/challenges/LSVRC/2012/index.php) dataset for model training
and validation.

### Training

<!--- Guideline: Please avoid using shell scripts in the command line. Python scripts preferred. -->

* Distributed Training

It is easy to reproduce the reported results with the pre-defined training recipe. For distributed training on multiple
Ascend 910 devices, please run

```shell
# distributed training on multiple NPU devices
msrun --bind_core=True --worker_num 8 python train.py --config configs/swintransformer/swin_tiny_ascend.yaml --data_dir /path/to/imagenet
```



For detailed illustration of all hyper-parameters, please refer
to [config.py](https://github.com/mindspore-lab/mindcv/blob/main/config.py).

**Note:**  As the global batch size  (batch_size x num_devices) is an important hyper-parameter, it is recommended to
keep the global batch size unchanged for reproduction or adjust the learning rate linearly to a new global batch size.

* Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on single NPU device
python train.py --config configs/swintransformer/swin_tiny_ascend.yaml --data_dir /path/to/dataset --distribute False
```

### Validation

To validate the accuracy of the trained model, you can use `validate.py` and parse the checkpoint path
with `--ckpt_path`.

```
python validate.py -c configs/swintransformer/swin_tiny_ascend.yaml --data_dir /path/to/imagenet --ckpt_path /path/to/ckpt
```

## Performance


Our reproduced model performance on ImageNet-1K is reported as follows.

Experiments are tested on ascend 910* with mindspore 2.3.1 graph mode.




| model name | params(M) | cards | batch size | resolution | jit level | graph compile | ms/step | img/s   | acc@top1 | acc@top5 | recipe                                                                                                  | weight                                                                                              |
| ---------- | --------- | ----- | ---------- | ---------- | --------- | ------------- | ------- | ------- | -------- | -------- | ------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| swin_tiny  | 33.38     | 8     | 256        | 224x224    | O2        | 266s          | 466.6   | 4389.20 | 80.90    | 94.90    | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/swintransformer/swin_tiny_ascend.yaml) | [weights](https://download-mindspore.osinfra.cn/toolkits/mindcv/swin/swin_tiny-72b3c5e6-910v2.ckpt) |



Experiments are tested on ascend 910 with mindspore 2.3.1 graph mode.




| model name | params(M) | cards | batch size | resolution | jit level | graph compile | ms/step | img/s   | acc@top1 | acc@top5 | recipe                                                                                                  | weight                                                                                |
| ---------- | --------- | ----- | ---------- | ---------- | --------- | ------------- | ------- | ------- | -------- | -------- | ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| swin_tiny  | 33.38     | 8     | 256        | 224x224    | O2        | 226s          | 454.49  | 4506.15 | 80.82    | 94.80    | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/swintransformer/swin_tiny_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/swin/swin_tiny-0ff2f96d.ckpt) |



### Notes

- top-1 and top-5: Accuracy reported on the validation set of ImageNet-1K.

## References

<!--- Guideline: Citation format GB/T 7714 is suggested. -->

[1] Liu Z, Lin Y, Cao Y, et al. Swin transformer: Hierarchical vision transformer using shifted windows[C]//Proceedings
of the IEEE/CVF international conference on computer vision. 2021: 10012-10022.
