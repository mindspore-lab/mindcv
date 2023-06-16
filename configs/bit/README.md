# BigTransfer

> [Big Transfer (BiT): General Visual Representation Learning](https://arxiv.org/abs/1912.11370)

## Introduction

Transfer of pre-trained representations improves sample efficiency and simplifies hyperparameter tuning when training deep neural networks for vision.
Big Transfer (BiT) can achieve strong performance on more than 20 data sets by combining some carefully selected components and using simple heuristic
methods for transmission. The components distilled by BiT for training models that transfer well are: 1) Big datasets: as the size of the dataset increases,
the optimal performance of the BIT model will also increase. 2) Big architectures: In order to make full use of large datasets, a large enough architecture
is required. 3) Long pre-training time: Pretraining on a larger dataset requires more training epoch and training time. 4) GroupNorm and Weight Standardisation:
BiT use GroupNorm combined with Weight Standardisation instead of BatchNorm. Since BatchNorm performs worse when the number of images on each accelerator is
too low. 5) With BiT fine-tuning, good performance can be achieved even if there are only a few examples of each type on natural images.[[1, 2](#References)]


## Results

Our reproduced model performance on ImageNet-1K is reported as follows.

<div align="center">

| Model          | Context  | Top-1 (%) | Top-5 (%) | Params(M) | Recipe                                                                                           |                                    Download                                 |
|----------------| -------- |-----------|-----------|-----------|--------------------------------------------------------------------------------------------------|  -------------------------------------------------------------------------- |
| bit_resnet50   | D910x8-G | 76.81     | 93.17     | 25.55     | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/bit/bit_resnet50_ascend.yaml)   | [weights](https://download.mindspore.cn/toolkits/mindcv/bit/BiT_resnet50-1e4795a4.ckpt) |
| bit_resnet50x3 | D910x8-G | 80.63     | 95.12     | 217.31    | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/bit/bit_resnet50x3_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/bit/BiT_resnet50x3-a960f91f.ckpt) |
| bit_resnet101  | D910x8-G | 77.93     | 93.75     | 44.54     | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/bit/bit_resnet101_ascend.yaml)  | [weights](https://download.mindspore.cn/toolkits/mindcv/bit/BiT_resnet101-2efa9106.ckpt) |

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
# distributed training on multiple GPU/Ascend devices
mpirun -n 8 python train.py --config configs/bit/bit_resnet50_ascend.yaml --data_dir /path/to/imagenet
```
> If the script is executed by the root user, the `--allow-run-as-root` parameter must be added to `mpirun`.

Similarly, you can train the model on multiple GPU devices with the above `mpirun` command.

For detailed illustration of all hyper-parameters, please refer to [config.py](https://github.com/mindspore-lab/mindcv/blob/main/config.py).

**Note:**  As the global batch size  (batch_size x num_devices) is an important hyper-parameter, it is recommended to keep the global batch size unchanged for reproduction or adjust the learning rate linearly to a new global batch size.

* Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on a CPU/GPU/Ascend device
python train.py --config configs/bit/bit_resnet50_ascend.yaml --data_dir /path/to/dataset --distribute False
```

### Validation

To validate the accuracy of the trained model, you can use `validate.py` and parse the checkpoint path with `--ckpt_path`.

```
python validate.py -c configs/bit/bit_resnet50_ascend.yaml --data_dir /path/to/imagenet --ckpt_path /path/to/ckpt
```

### Deployment

Please refer to the [deployment tutorial](https://mindspore-lab.github.io/mindcv/tutorials/deployment/) in MindCV.

## References

<!--- Guideline: Citation format should follow GB/T 7714. -->
[1] Kolesnikov A, Beyer L, Zhai X, et al. Big transfer (bit): General visual representation learning[C]//European conference on computer vision. Springer, Cham, 2020: 491-507.

[2] BigTransfer (BiT): State-of-the-art transfer learning for computer vision, https://blog.tensorflow.org/2020/05/bigtransfer-bit-state-of-art-transfer-learning-computer-vision.html
