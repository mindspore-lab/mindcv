# PNASNet(Progressive Neural Architecture Search Network)
> For more details, please refer to: [Progressive Neural Architecture Search](https://arxiv.org/abs/1712.00559)

## Introduction

<div align=center>

![](bestarchitecture.png)![](modelarchitecture.png)
</div>

The key idea of PNASNet is that the design of sequential model-based optimation to reduce the search complexity. In addition, some operations in search pool are abandoned in PNASNet to reduce the search space. Besides that, a predictor is utilized to predict the accuracy instead of training every model. The left figure shows that the best cell architecture after search. The right figure shows the model architecture of PNASNet. PNASNet could achieve better model performance with faster search speed on [ImageNet-1K dataset](https://www.image-net.org/download.php) compared with NASNet.

## Results

| Model           | Context   |  Top-1 (%)  | Top-5 (%)  |  Params (M)    | Train T. | Infer T. |  Download | Config | Log |
|-----------------|-----------|-------|-------|------------|-------|--------|---|--------|--------------|
| PNASNet-5 | D910x8-G | -     | -     | -       | -s/epoch | -ms/step | [model]() | [cfg]() | [log]() |


#### Notes

- All models are trained on ImageNet-1K training set and the top-1 accuracy is reported on the validatoin set.
- Context: GPU_TYPE x pieces - G/F, G - graph mode, F - pynative mode with ms function.  

## Quick Start
<details>
<summary>Preparation</summary>

#### Installation
Please refer to the [installation instruction](https://github.com/mindspore-ecosystem/mindcv#installation) in MindCV.

#### Dataset Preparation
Please download the [ImageNet-1K](https://www.image-net.org/download.php) dataset for model training and validation.
</details>

<details>
<summary>Training</summary>

- **Hyper-parameters.** The hyper-parameter configurations for producing the reported results are stored in the yaml files in `mindcv/configs/pnasnet` folder. For example, to train with one of these configurations, you can run:

  ```shell
  # train PNASNet-5 on 8 GPUs
  mpirun -n 8 python train.py --config path/to/pnasnet/yaml/file --data_dir /path/to/imagenet
  ```

  Note that the number of GPUs/Ascends and batch size will influence the training results. To reproduce the training result at most, it is recommended to use the **same number of GPUs/Ascends** with the same batch size.

Detailed adjustable parameters and their default value can be seen in [config.py](../../config.py).
</details>

<details>
<summary>Validation</summary>

- To validate the model, you can use `validate.py`. Here is an example for PNASNet-5 to verify the accuracy of your training.

  ```shell
  python validate.py --config path/to/pnasnet/yaml/file --data_dir /path/to/imagenet --ckpt_path /path/to/pnasnet/file.ckpt
  ```
</details>

<details>
<summary>Deployment (optional)</summary>

Please refer to the deployment tutorial in MindCV.
</details>


