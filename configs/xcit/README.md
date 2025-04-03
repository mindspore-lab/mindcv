# XCiT: Cross-Covariance Image Transformers

> [XCiT: Cross-Covariance Image Transformers](https://arxiv.org/abs/2106.09681)



## Introduction

XCiT models propose a “transposed” version of self-attention that operates across feature channels rather than tokens,
where the interactions are based on the cross-covariance matrix between keys and queries. The resulting cross-covariance
attention (XCA) has linear complexity in the number of tokens, and allows efficient processing of high-resolution
images. Our cross-covariance image transformer (XCiT) – built upon XCA – combines the accuracy of conventional
transformers with the scalability of convolutional architectures.

<p align="center">
  <img src="https://user-images.githubusercontent.com/51260139/211969416-b57b3aff-49b0-4048-b970-55d9196ed63b.png" width=600 />
</p>
<p align="center">
  <em>Figure 1. Architecture of XCiT [<a href="#references">1</a>] </em>
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

Please download the [ImageNet-1K](https://www.image-net.org/challenges/LSVRC/2012/index.php) dataset for model training
and validation.

### Training

* Distributed Training

It is easy to reproduce the reported results with the pre-defined training recipe. For distributed training on multiple
Ascend 910 devices, please run

```shell
# distributed training on multiple NPU devices
msrun --bind_core=True --worker_num 8 python train.py --config configs/xcit/xcit_tiny_12_p16_ascend.yaml --data_dir /path/to/imagenet
```


For detailed illustration of all hyper-parameters, please refer
to [config.py](https://github.com/mindspore-lab/mindcv/blob/main/config.py).

**Note:**  As the global batch size  (batch_size x num_devices) is an important hyper-parameter, it is recommended to
keep the global batch size unchanged for reproduction or adjust the learning rate linearly to a new global batch size.

* Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on single NPU device
python train.py --config configs/xcit/xcit_tiny_12_p16_ascend.yaml --data_dir /path/to/dataset --distribute False
```

### Validation

To validate the accuracy of the trained model, you can use `validate.py` and parse the checkpoint path
with `--ckpt_path`.

```
python validate.py -c configs/xcit/xcit_tiny_12_p16_224_ascend.yaml --data_dir /path/to/imagenet --ckpt_path /path/to/ckpt
```
## Performance

Our reproduced model performance on ImageNet-1K is reported as follows.

Experiments are tested on ascend 910* with mindspore 2.5.0 graph mode.

| model name  | cards | batch size | resolution | jit level | graph compile | ms/step | img/s   | recipe                                                                                           | weight                                                                                                  | acc@top1 | acc@top5 |
| ----------- | ----- | ---------- | ---------- | --------- | ------------- | ------- | ------- | ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------- | -------- | -------- |
| xcit       | 8     | 128        | 224x224    | O2        | 329s          | 233.86  | 4378.69 | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/xcit/xcit_tiny_12_p16_224_ascend.yaml)           | [weights](https://download-mindspore.osinfra.cn/toolkits/mindcv/xcit/xcit_tiny_12_p16_224-bd90776e-910v2.ckpt)           | 77.16    | 93.57    |

### Notes

- top-1 and top-5: Accuracy reported on the validation set of ImageNet-1K.


## References

<!--- Guideline: Citation format should follow GB/T 7714. -->
[1] Ali A, Touvron H, Caron M, et al. Xcit: Cross-covariance image transformers[J]. Advances in neural information
processing systems, 2021, 34: 20014-20027.
