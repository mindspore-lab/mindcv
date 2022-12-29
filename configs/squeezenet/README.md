# SqueezeNet

> [SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size](https://arxiv.org/abs/1602.07360)

## Introduction

SqueezeNet is a smaller CNN architectures which is comprised mainly of Fire modules and it achieves AlexNet-level
accuracy on ImageNet with 50x fewer parameters. SqueezeNet can offer at least three advantages: (1) Smaller CNNs require
less communication across servers during distributed training. (2) Smaller CNNs require less bandwidth to export a new
model from the cloud to an autonomous car. (3) Smaller CNNs are more feasible to deploy on FPGAs and other hardware with
limited memory. Additionally, with model compression techniques, SqueezeNet is able to be compressed to less than
0.5MB (510× smaller than AlexNet). Blow is macroarchitectural view of SqueezeNet architecture. Left: SqueezeNet ;
Middle: SqueezeNet with simple bypass; Right: SqueezeNet with complex bypass.

<p align="center">
  <img src="https://github.com/mindspore-lab/mindcv/blob/main/configs/squeezenet/squeezenet.png" width=800 />  
</p>
<p align="center">
  <em>Figure 1. Architecture of SqueezeNet [<a href="#references">1</a>] </em>
</p>

## Results

<div align="center">

| Model           | Context   |  Top-1 (%) | Top-5 (%)  |  Params (M) | Recipe  | Download |
|-----------------|-----------|------------|------------|-------------|---------|----------|
| squeezenet_1.0 | GPUx8-G     | 59.49      | 81.22     |    1.25  | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/squeezenet/squeezenet_1.0_gpu.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/squeezenet/squeezenet_1.0_224.ckpt)  |
| squeezenet_1.1 | GPUx8-G      | 58.99      | 80.99    |    1.24    | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/squeezenet/squeezenet_1.1_gpu.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/squeezenet/squeezenet_1.1_224.ckpt)  |

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
<!--- Guideline: Avoid using shell script in the command line. Python script preferred. -->

* Distributed Training

It is easy to reproduce the reported results with the pre-defined training recipe. For distributed training on multiple Ascend 910 devices, please run

```shell
# distrubted training on multiple GPU/Ascend devices
mpirun -n 8 python train.py --config configs/squeezenet/squeezenet_1.0_gpu.yaml --data_dir /path/to/imagenet
```
  
Similarly, you can train the model on multiple GPU devices with the above `mpirun` command.

For detailed illustration of all hyper-parameters, please refer to [config.py](https://github.com/mindspore-lab/mindcv/blob/main/config.py).

**Note:**  As the global batch size  (batch_size x num_devices) is an important hyper-parameter, it is recommended to keep the global batch size unchanged for reproduction or adjust the learning rate linearly to a new global batch size.

* Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on a CPU/GPU/Ascend device
python train.py --config configs/squeezenet/squeezenet_1.0_gpu.yaml --data_dir /path/to/dataset --distribute False
```

### Validation

To validate the accuracy of the trained model, you can use `validate.py` and parse the checkpoint path with `--ckpt_path`.

```
python validate.py -c configs/squeezenet/squeezenet_1.0_gpu.yaml --data_dir /path/to/imagenet --ckpt_path /path/to/ckpt
```

### Deployment

To deploy online inference services with the trained model efficiently, please refer to the [deployment tutorial](https://github.com/mindspore-lab/mindcv/blob/main/tutorials/deployment.md).


### References
<!--- Guideline: Citation format GB/T 7714 is suggested. -->
[1] Iandola F N, Han S, Moskewicz M W, et al. SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and< 0.5 MB model size[J]. arXiv preprint arXiv:1602.07360, 2016.

