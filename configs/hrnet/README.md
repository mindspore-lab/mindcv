# HRNet
<!--- Guideline: use url linked to abstract in ArXiv instead of PDF for fast loading.  -->

> [Deep High-Resolution Representation Learning for Visual Recognition](https://arxiv.org/abs/1908.07919)

## Introduction
<!--- Guideline: Introduce the model and architectures. Cite if you use/adopt paper explanation from others. -->

High-resolution representations are essential for position-sensitive vision problems, such as human pose estimation, semantic segmentation, and object detection. Existing state-of-the-art frameworks first encode the input image as a low-resolution representation through a subnetwork that is formed by connecting high-to-low resolution convolutions (e.g., ResNet, VGGNet), and then recover the high-resolution representation from the encoded low-resolution representation. Instead, the proposed network, named as High-Resolution Network (HRNet), maintains high-resolution representations through the whole process. There are two key characteristics: (i) Connect the high-to-low resolution convolution streams in parallel; (ii) Repeatedly exchange the information across resolutions. The benefit is that the resulting representation is semantically richer and spatially more precise. It shows the superiority of the proposed HRNet in a wide range of applications, including human pose estimation, semantic segmentation, and object detection, suggesting that the HRNet is a stronger backbone for computer vision problems.

<!--- Guideline: If an architecture table/figure is available in the paper, put one here and cite for intuitive illustration. -->

<p align="center">
  <img src="https://user-images.githubusercontent.com/8342575/218354682-4256e17e-bb69-4e51-8bb9-a08fc29087c4.png" width=800 />
</p>
<p align="center">
  <em> Figure 1. Architecture of HRNet [<a href="#references">1</a>] </em>
</p>

## Results
<!--- Guideline:
Table Format:
- Model: model name in lower case with _ seperator.
- Top-1 and Top-5: Keep 2 digits after the decimal point.
- Params (M): # of model parameters in millions (10^6). Keep 2 digits after the decimal point
- Recipe: Training recipe/configuration linked to a yaml config file. Use absolute url path.
- Download: url of the pretrained model weights. Use absolute url path.
-->

Our reproduced model performance on ImageNet-1K is reported as follows.

- ascend 910* with graph mode

<div align="center">


| model     | top-1 (%) | top-5 (%) | params (M) | batch size | cards | ms/step | jit_level | recipe                                                                                        | download                                                                                             |
| --------- | --------- | --------- | ---------- | ---------- | ----- | ------- | --------- | --------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| hrnet_w32 | 80.66     | 95.30     | 41.30      | 128        | 8     | 238.03  | O2        | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/hrnet/hrnet_w32_ascend.yaml) | [weights](https://download-mindspore.osinfra.cn/toolkits/mindcv/hrnet/hrnet_w32-e616cdcb-910v2.ckpt) |



</div>

- ascend 910 with graph mode

<div align="center">


| model     | top-1 (%) | top-5 (%) | params (M) | batch size | cards | ms/step | jit_level | recipe                                                                                        | download                                                                               |
| --------- | --------- | --------- | ---------- | ---------- | ----- | ------- | --------- | --------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| hrnet_w32 | 80.64     | 95.44     | 41.30      | 128        | 8     | 279.10  | O2        | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/hrnet/hrnet_w32_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/hrnet/hrnet_w32-cc4fbd91.ckpt) |



</div>

#### Notes
- Top-1 and Top-5: Accuracy reported on the validation set of ImageNet-1K.


## Quick Start
### Preparation

#### Installation
Please refer to the [installation instruction](https://mindspore-lab.github.io/mindcv/installation/) in MindCV.

#### Dataset Preparation
Please download the [ImageNet-1K](https://www.image-net.org/challenges/LSVRC/2012/index.php) dataset for model training and validation.

### Training
<!--- Guideline: Avoid using shell script in the command line. Python script preferred. -->

* Distributed Training

It is easy to reproduce the reported results with the pre-defined training recipe. For distributed training on multiple Ascend 910 devices, please run

```shell
# distributed training on multiple NPU devices
msrun --bind_core=True --worker_num 8 python train.py --config configs/hrnet/hrnet_w32_ascend.yaml --data_dir /path/to/imagenet
```



For detailed illustration of all hyper-parameters, please refer to [config.py](https://github.com/mindspore-lab/mindcv/blob/main/config.py).

**Note:**  As the global batch size  (batch_size x num_devices) is an important hyper-parameter, it is recommended to keep the global batch size unchanged for reproduction or adjust the learning rate linearly to a new global batch size.

* Standalone Training

If you want to train or finetune the model on a smaller dataset without distributed training, please run:

```shell
# standalone training on single NPU device
python train.py --config configs/hrnet/hrnet_w32_ascend.yaml --data_dir /path/to/dataset --distribute False
```

### Validation

To validate the accuracy of the trained model, you can use `validate.py` and parse the checkpoint path with `--ckpt_path`.

```
python validate.py -c configs/hrnet/hrnet_w32_ascend.yaml --data_dir /path/to/imagenet --ckpt_path /path/to/ckpt
```

## References
<!--- Guideline: Citation format GB/T 7714 is suggested. -->

[1] Jingdong Wang, Ke Sun, Tianheng Cheng, et al. Deep High-Resolution Representation Learning for Visual Recognition[J]. arXiv preprint arXiv:1908.07919, 2019.
