# RepMLPNet
> [RepMLPNet: Hierarchical Vision MLP with Re-parameterized Locality](https://arxiv.org/pdf/2112.11081v2.pdf)

## Introduction
***

Accepted to CVPR-2022!

The latest version: https://openaccess.thecvf.com/content/CVPR2022/papers/Ding_RepMLPNet_Hierarchical_Vision_MLP_With_Re-Parameterized_Locality_CVPR_2022_paper.pdf

Compared to the old version, we no longer use RepMLP Block as a plug-in component in traditional ConvNets. Instead, we build an MLP architecture with RepMLP Block with a hierarchical design. RepMLPNet shows favorable performance, compared to the other vision MLP models including MLP-Mixer, ResMLP, gMLP, S2-MLP, etc.

Of course, you may also use it in your model as a building block.

The overlap between the two versions is the Structural Re-parameterization method (Localtiy Injection) that equivalently merges conv into FC. The architectural designs presented in the latest version significantly differ from the old version (ResNet-50 + RepMLP).

![](repmlpblock.png)

## Results
***

| Model           | Context   |  Top-1 (%)  | Top-5 (%)  |  Params (M)    | Train T. | Infer T. |  Download | Config | Log |
|-----------------|-----------|-------|-------|------------|-------|--------|---|--------|--------------|
| repmlp_t224 | D910x8 | 76.649     |      | 38.3M       | 1011s/epoch | 15.8ms/step | [model]() | [cfg]() | [log]() |


#### Notes

- All models are trained on ImageNet-1K training set and the top-1 accuracy is reported on the validatoin set.
- Context: GPU_TYPE x pieces - G/F, G - graph mode, F - pynative mode with ms function.  

## Quick Start
***
### Preparation

#### Installation
Please refer to the [installation instruction](https://github.com/mindspore-ecosystem/mindcv#installation) in MindCV.

#### Dataset Preparation
Please download the [ImageNet-1K](https://www.image-net.org/download.php) dataset for model training and validation.

### Training

- **Hyper-parameters.** The hyper-parameter configurations for producing the reported results are stored in the yaml files in `mindcv/configs/repmlp` folder. For example, to train with one of these configurations, you can run:

  ```shell
  # train repmlp_t224 on 8 Ascends
  bash ./scripts/run_distribution_ascend.sh ./scripts/rank_table_8pcs.json [DATASET_PATH] ./config/repmlp/repmlp_T224.yaml
  ```
  
  Note that the number of GPUs/Ascends and batch size will influence the training results. To reproduce the training result at most, it is recommended to use the **same number of GPUs/Ascneds** with the same batch size.


Detailed adjustable parameters and their default value can be seen in [config.py](../../config.py).

### Validation

- To validate the trained model, you can use `validate.py`. Here is an example for densenet121 to verify the accuracy of
  pretrained weights.

  ```shell
  python validate.py --model=RepMLPNet_T224 --data_dir=imagenet_dir --val_split=val --ckpt_path
  ```


### Deployment (optional)

Please refer to the deployment tutorial in MindCV.



