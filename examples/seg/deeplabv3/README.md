# DeepLabV3 Based on MindCV Backbones

> [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587)

## Introduction

DeepLabV3 is a semantic segmentation architecture improved over previous version. Two main contributions of DeepLabV3 are as follows. 1) Modules are designed which employ atrous convolution in cascade or in parallel to capture multi-scale context by adopting multiple atrous rates to handle the problem of segmenting objects at multiple scale. 2) The Atrous Spatial Pyramid Pooling (ASPP) module is augmented with image-level features encoding global context and further boost performance. The improved ASPP applys global average pooling on the last feature map of the model, feeds the resulting image-level features to a 1 × 1 convolution with 256 filters (and batch normalization), and then bilinearly upsamples the feature to the desired spatial dimension. The DenseCRF post-processing from DeepLabv2 is deprecated.

<p align="center">
  <img src="https://github.com/mindspore-lab/mindcv/assets/33061146/db2076ed-bccd-455f-badb-e03deb131dc5" width=800 />
</p>
<p align="center">
  <em>Figure 1. Architecture of DeepLabV3 with output_stride=16 [<a href="#references">1</a>] </em>
</p>

This example provides an implementation of DeepLabV3 using backbones from MindCV. More details about feature extraction of MindCV are in [this tutorial](https://github.com/mindspore-lab/mindcv/blob/main/docs/en/how_to_guides/feature_extraction.md). Note that the ResNet in DeeplabV3 contains atrous convolutions with different rates,  `dilated_resnet.py`  is provided as a modification of ResNet from MindCV, with atrous convolutions in block 3-4.

## Quick Start

### Preparation

1. Clone MindCV repository by running

   ```shell
   git clone https://github.com/mindspore-lab/mindcv.git
   ```
2. Install dependencies as shown [here](https://mindspore-lab.github.io/mindcv/installation/)

3. Prepare dataset

- Download Pascal VOC 2012 dataset,  [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) and Semantic Boundaries Dataset, [SBD](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz).

- Prepare training and test data list files with the path to image and annotation pairs. You could simply run `python examples/seg/deeplabv3/build_dataset/get_dataset_lst.py --data_root=[PATH TO DATA]` to generate the list files. The lines in list file should be like as follows:

  ```
  /[PATH..]/JPEGImages/2007_000032.jpg /[PATH..]/SegmentationClassGray/2007_000032.png
  /[PATH..]/JPEGImages/2007_000039.jpg /[PATH..]/SegmentationClassGray/2007_000039.png
  /[PATH..]/JPEGImages/2007_000063.jpg /[PATH..]/SegmentationClassGray/2007_000063.png
  ......
  ```

- Convert training dataset to mindrecords by running  ``bulid_seg_data.py `` script.

  ```shell
  python examples/seg/deeplabv3/bulid_dataset/bulid_seg_data.py \
  		--data_root=[root path of training data] \
  		--data_lst=[path of data list file prepared above] \
  		--dst_path=[path to save mindrecords] \
  		--num_shards=8 \
  		--shuffle=True
  ```

* In accord with paper, we train on *trainaug* dataset (voc train + SBD) and evaluate on *voc val* dataset.

4. Backbone: download pre-trained backbone from MindCV, here we use [ResNet101](https://download.mindspore.cn/toolkits/mindcv/resnet/resnet101-689c5e77.ckpt).

### Train

It is highly recommended to use **distributed training** for this DeepLabV3 implementation.

For distributed training using **OpenMPI's `mpirun`**, simply run
```shell
cd mindcv  # change directory to the root of MindCV repository
mpirun -n [# of devices] python examples/seg/deeplabv3/train.py --config [the path to the config file]
```

For distributed training with [Ascend rank table](https://github.com/mindspore-lab/mindocr/blob/main/docs/en/tutorials/distribute_train.md#12-configure-rank_table_file-for-training), configure `ascend8p.sh` as follows

```shell
#!/bin/bash
export DEVICE_NUM=8
export RANK_SIZE=8
export RANK_TABLE_FILE="./hccl_8p_01234567_127.0.0.1.json"

for ((i = 0; i < ${DEVICE_NUM}; i++)); do
   export DEVICE_ID=$i
   export RANK_ID=$i
   python -u examples/seg/deeplabv3/train.py --config [the path to the config file]  &> ./train_$i.log &
done
```

and start training by running:
```shell l
cd mindcv  # change directory to the root of MindCV repository
bash ascend8p.sh
```

For single-device training, simply set the parameter ``distributed`` to ``False`` in config file and run:
```shell
cd mindcv  # change directory to the root of MindCV repository
python examples/seg/deeplabv3/train.py --config [the path to the config file]
```

##### The training steps are as follow:

- Step 1: Employ output_stride=16 and fine-tune pretrained resnet101 on *trainaug* dataset. In config file, please specify the path of pretrained backbone checkpoint in keyword `backbone_ckpt_path` and set `output_stride` to `16`.

  ```shell
  mpirun -n 8 python examples/seg/deeplabv3/train.py --config examples/seg/deeplabv3/deeplabv3_s16_dilated_resnet101.yaml
  ```

* Step 2: Employ output_stride=8, fine-tune model from step 1 on  *trainaug* dataset with smaller base learning rate. In config file, please specify the path of checkpoint from previous step in `ckpt_path`, set  `ckpt_pre_trained` to `True` and set `output_stride` to `8` .

  ```shell
  mpirun -n 8 python examples/seg/deeplabv3/train.py --config examples/seg/deeplabv3/deeplabv3_s8_dilated_resnet101.yaml
  ```

### Test

For testing the trained model, first specify the path to the model checkpoint at keyword `ckpt_path` in the config file. You could modify `output_stride`, `flip`, `scales` in config file during inference.
```shell
cd mindcv  # change directory to the root of MindCV repository
python examples/seg/deeplabv3/eval.py --config [the path to the config file]
```
For example, after replacing  `ckpt_path` in config file with [checkpoint](https://download.mindspore.cn/toolkits/mindcv/deeplabv3/deeplabv3_s8_resnet101-a297e7af.ckpt) from 2-step training, commands below employ os=8 without left-right filpped or muticale inputs.
```shell
cd mindcv  # change directory to the root of MindCV repository
python examples/det/ssd/eval.py --config examples/seg/deeplabv3/deeplabv3_s8_dilated_resnet101.yaml
```

## Results



### Model results


| Train OS | Infer OS |  MS  | FLIP | mIoU  |                            Config                            |                           Download                           |
| :------: | :------: | :--: | :--: | :---: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|    16    |    16    |      |      | 78.20 | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/examples/seg/deeplabv3/deeplabv3_s16_dilated_resnet101.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/deeplabv3/deeplabv3_s16_resnet101-9de3c664.ckpt) |
|  16, 8   |    16    |      |      | 77.33 | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/examples/seg/deeplabv3/deeplabv3_s16_dilated_resnet101.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/deeplabv3/deeplabv3_s8_resnet101-a297e7af.ckpt) |
|  16, 8   |    8     |      |      | 79.16 | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/examples/seg/deeplabv3/deeplabv3_s8_dilated_resnet101.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/deeplabv3/deeplabv3_s8_resnet101-a297e7af.ckpt) |
|  16, 8   |    8     |  √   |      | 79.93 | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/examples/seg/deeplabv3/deeplabv3_s8_dilated_resnet101.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/deeplabv3/deeplabv3_s8_resnet101-a297e7af.ckpt) |
|  16, 8   |    8     |  √   |  √   | 80.14 | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/examples/seg/deeplabv3/deeplabv3_s8_dilated_resnet101.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/deeplabv3/deeplabv3_s8_resnet101-a297e7af.ckpt) |

**Note**: **OS**: output stride.  **MS**: multiscale inputs during test. **Flip**: adding left-right flipped inputs during test. **Train OS = 16** means training step 1 mentioned in train scetion above, and **Train OS = 16, 8** means the entire two-step training.

As illustrated in paper, adding left-right flipped inputs or muilt-scale inputs during test could improve the performence. Also, once the model is finally trained, employed output_stride=8 during inference bring improvement over using  output_stride=16.


## References
[1] Chen L C, Papandreou G, Schroff F, et al. Rethinking atrous convolution for semantic image segmentation[J]. arXiv preprint arXiv:1706.05587, 2017.
