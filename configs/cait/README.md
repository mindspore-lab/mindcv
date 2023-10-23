# Going deeper with Image Transformers

> [Going deeper with Image Transformers](https://arxiv.org/abs/2103.17239)

## Introduction

CaiT is built based on ViT but made two contributions to improve model performance.
Firstly, Layerscale is introduced to facilitate the convergence.
Secondly, class-attention offers a more effective processing of the class embedding.
By combing these parts, Cait could get a SOTA performance on ImageNet-1K dataset.


## Results

Our reproduced model performance on ImageNet-1K is reported as follows.

<div align="center">

| Model          | Context  | Top-1 (%) | Top-5 (%) | Params(M) | Recipe                                                                                     | Download                                                                               |
|----------------| -------- |----------|-----------|-----------|--------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------|
| cait_xxs24_224 | D910x8-G | 77.71    | 94.10     | 11.94     | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/cait/cait_xxs24_224.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/cait/cait_xxs24-31b307a8.ckpt) |
| cait_xs24_224  | D910x8-G | 81.29    | 95.60     | 26.53     | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/cait/cait_xs24_224.yaml)  | [weights](https://download.mindspore.cn/toolkits/mindcv/cait/cait_xs24-ba0c2053.ckpt)  |
| cait_s24_224   | D910x8-G | 82.25    | 95.95     | 46.88     | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/cait/cait_s24_224.yaml)   | [weights](https://download.mindspore.cn/toolkits/mindcv/cait/cait_s24-0a06be71.ckpt)   |
| cait_s36_224   | D910x8-G | 82.11    | 95.84     | 68.16     | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/cait/cait_s36_224.yaml)   | [weights](https://download.mindspore.cn/toolkits/mindcv/cait/cait_s36-2e42bfc8.ckpt)   |


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
mpirun -n 8 python train.py --config configs/cait/cait_xxs24_224.yaml --data_dir /path/to/imagenet
```
> If the script is executed by the root user, the `--allow-run-as-root` parameter must be added to `mpirun`.

Similarly, you can train the model on multiple GPU devices with the above `mpirun` command.

For detailed illustration of all hyper-parameters, please refer to [config.py](https://github.com/mindspore-lab/mindcv/blob/main/config.py).

**Note:**  As the global batch size  (batch_size x num_devices) is an important hyper-parameter, it is recommended to keep the global batch size unchanged for reproduction or adjust the learning rate linearly to a new global batch size.

* Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on a CPU/GPU/Ascend device
python train.py --config configs/cait/cait_xxs24_224.yaml --data_dir /path/to/dataset --distribute False
```

### Validation

To validate the accuracy of the trained model, you can use `validate.py` and parse the checkpoint path with `--ckpt_path`.

```
python validate.py -c configs/cait/cait_xxs24_224.yaml --data_dir /path/to/imagenet --ckpt_path /path/to/ckpt
```

### Deployment

Please refer to the [deployment tutorial](https://mindspore-lab.github.io/mindcv/tutorials/deployment/).

## References

<!--- Guideline: Citation format should follow GB/T 7714. -->
[1] Touvron H, Cord M, Sablayrolles A, et al. Going deeper with image transformers[C]//Proceedings of the IEEE/CVF international conference on computer vision. 2021: 32-42.
