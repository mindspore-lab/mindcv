# XCiT: Cross-Covariance Image Transformers

> [XCiT: Cross-Covariance Image Transformers](https://arxiv.org/abs/2106.09681)

## Introduction

XCiT models propose a “transposed” version of self-attention that operates across feature channels rather than tokens, where the interactions are based on the cross-covariance matrix between keys and queries. The resulting cross-covariance attention (XCA) has linear complexity in the number of tokens, and allows efficient processing of high-resolution images. Our cross-covariance image transformer (XCiT) – built upon XCA – combines the accuracy of conventional transformers with the scalability of convolutional architectures.

<p align="center">
  <img src="https://user-images.githubusercontent.com/51260139/211969416-b57b3aff-49b0-4048-b970-55d9196ed63b.png" width=600 />
</p>
<p align="center">
  <em>Figure 1. Architecture of XCiT [<a href="#references">1</a>] </em>
</p>


* **Linear complexity in time and memory**

  XCiT models has a linear complexity w.r.t number of patches/tokens: $\mathcal{O}(N d ^2)$


![](https://user-images.githubusercontent.com/51260139/211969388-0658c89b-c41c-4df9-a295-5b3431b626b7.png)  |  ![](https://user-images.githubusercontent.com/51260139/211969950-92b15d1d-0b08-4075-9a12-faf40cd49efa.png)
:-------------------------:|:-------------------------:
Peak Memory (inference)    | Millisecond/Image (Inference)


## Results

Our reproduced model performance on ImageNet-1K is reported as follows.

<div align="center">

| Model        | Context  | Top-1 (%) | Top-5 (%) | Params (M) | Recipe                                                                                        | Download                                                                       |
|--------------|----------|-----------|-----------|------------|-----------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------|
| xcit_tiny_12_p16 | D910x8-G | 77.674     | 93.792      | 7      | [yaml](https://github.com/bnucsy/mindcv/tree/main/configs/xcit/xcit_tiny_12_p16_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/xcit/xcit_tiny_12_p16_best.ckpt) |

</div>


#### Notes

- Context: Training context denoted as {device}x{pieces}-{MS mode}, where mindspore mode can be G - graph mode or F - pynative mode with ms function. For example, D910x8-G is for training on 8 pieces of Ascend 910 NPU using graph mode.
- Top-1 and Top-5: Accuracy reported on the validation set of ImageNet-1K.

## Quick Start

### Preparation

#### Installation

Please refer to the [installation instruction](https://github.com/mindspore-lab/mindcv#installation) in MindCV.

#### Dataset Preparation

Please download the [ImageNet-1K](https://www.image-net.org/challenges/LSVRC/2012/index.php) dataset for model training and validation.

### Training

* Distributed Training

It is easy to reproduce the reported results with the pre-defined training recipe. For distributed training on multiple Ascend 910 devices, please run

```shell
# distrubted training on multiple GPU/Ascend devices
mpirun -n 8 python train.py --config configs/xcit/xcit_tiny_12_p16_ascend.yaml --data_dir /path/to/imagenet
```
> If the script is executed by the root user, the `--allow-run-as-root` parameter must be added to `mpirun`.

Similarly, you can train the model on multiple GPU devices with the above `mpirun` command.

For detailed illustration of all hyper-parameters, please refer to [config.py](https://github.com/mindspore-lab/mindcv/blob/main/config.py).

**Note:**  As the global batch size  (batch_size x num_devices) is an important hyper-parameter, it is recommended to keep the global batch size unchanged for reproduction or adjust the learning rate linearly to a new global batch size.

* Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on a CPU/GPU/Ascend device
python train.py --config configs/xcit/xcit_tiny_12_p16_ascend.yaml --data_dir /path/to/dataset --distribute False
```

### Validation

To validate the accuracy of the trained model, you can use `validate.py` and parse the checkpoint path with `--ckpt_path`.

```
python validate.py -c configs/xcit/xcit_tiny_12_p16_ascend.yaml --data_dir /path/to/imagenet --ckpt_path /path/to/ckpt
```

### Deployment

Please refer to the [deployment tutorial](https://github.com/mindspore-lab/mindcv/blob/main/tutorials/deployment.md) in MindCV.

## References

<!--- Guideline: Citation format should follow GB/T 7714. -->
[1] Ali A, Touvron H, Caron M, et al. Xcit: Cross-covariance image transformers[J]. Advances in neural information processing systems, 2021, 34: 20014-20027.
