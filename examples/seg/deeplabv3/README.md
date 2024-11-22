# DeepLabV3, DeeplabV3+ Based on MindCV Backbones

> DeeplabV3: [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587)
>
> DeeplabV3+:[Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611)


## Introduction

**DeepLabV3** is a semantic segmentation architecture improved over previous version. Two main contributions of DeepLabV3 are as follows. 1) Modules are designed which employ atrous convolution in cascade or in parallel to capture multi-scale context by adopting multiple atrous rates to handle the problem of segmenting objects at multiple scale. 2) The Atrous Spatial Pyramid Pooling (ASPP) module is augmented with image-level features encoding global context and further boost performance. The improved ASPP applys global average pooling on the last feature map of the model, feeds the resulting image-level features to a 1 × 1 convolution with 256 filters (and batch normalization), and then bilinearly upsamples the feature to the desired spatial dimension. The DenseCRF post-processing from DeepLabV2 is deprecated.

<p align="center">
  <img src="https://github.com/mindspore-lab/mindcv/assets/33061146/db2076ed-bccd-455f-badb-e03deb131dc5" width=700/>
</p>
<p align="center">
  <em>Figure 1. Architecture of DeepLabV3 with output_stride=16 [<a href="#references">1</a>] </em>
</p>



**DeepLabV3+** extends DeepLabv3 by adding a simple yet effective decoder module to refine the segmentation results especially along object boundaries. It combines advantages from Spatial pyramid pooling module and encode-decoder structure. The last feature map before logits in the origin deeplabv3 becomes the encoder output.  The encoder features are first bilinearly upsampled by a factor of 4 and then concatenated with the corresponding low-level features  from the network backbone that have the same spatial resolution. Another 1 × 1 convolution is applied on the low-level features to reduce the number of channels. After the concatenation, a few 3 × 3 convolutions are applied to refine the features followed by another simple bilinear upsampling by a factor of 4.

<p align="center">
  <img src="https://github.com/mindspore-lab/mindcv/assets/33061146/e1a17518-b19a-46f1-b28a-ec67cafa81be" width=700/>
</p>
<p align="center">
  <em>Figure 2. DeepLabv3+ extends DeepLabv3 by employing a encoderdecoder structure [<a href="#references">2</a>] </em>
</p>


This example provides implementations of DeepLabV3 and DeepLabV3+ using backbones from MindCV. More details about feature extraction of MindCV are in [this tutorial](https://github.com/mindspore-lab/mindcv/blob/main/docs/en/how_to_guides/feature_extraction.md). Note that the ResNet in DeepLab contains atrous convolutions with different rates,  `dilated_resnet.py`  is provided as a modification of ResNet from MindCV, with atrous convolutions in block 3-4.

## Requirements
| mindspore | ascend driver |  firmware   | cann toolkit/kernel |
| :-------: | :-----------: | :---------: | :-----------------: |
|   2.3.1   |   24.1.RC2    | 7.3.0.1.231 |    8.0.RC2.beta1    |

## Quick Start

### Preparation

1. Clone MindCV repository, enter `mindcv`  and assume we are always in this project root.

   ```shell
   git clone https://github.com/mindspore-lab/mindcv.git
   cd mindcv
   ```

2. Install dependencies as shown [here](https://mindspore-lab.github.io/mindcv/installation/), and also install `cv2`, `addict`.

   ```shell
   pip install opencv-python
   pip install addict
   ```

3. Prepare dataset

   * Download Pascal VOC 2012 dataset,  [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) and Semantic Boundaries Dataset, [SBD](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz).

   * Prepare training and test data list files with the path to image and annotation pairs. You could simply run `python examples/seg/deeplabv3/preprocess/get_data_list.py --data_root=/path/to/data` to generate the list files. This command results in 5 data list files. The lines in a list file should be like as follows:

     ```
     /path/to/data/JPEGImages/2007_000032.jpg /path/to/data/SegmentationClassGray/2007_000032.png
     /path/to/data/JPEGImages/2007_000039.jpg /path/to/data/SegmentationClassGray/2007_000039.png
     /path/to/data/JPEGImages/2007_000063.jpg /path/to/data/SegmentationClassGray/2007_000063.png
     ......
     ```

   * Convert training dataset to mindrecords by running  ``build_seg_data.py`` script. In accord with paper, we train on *trainaug* dataset (*voc train* + *SBD*). You can train on other dataset by changing the data list path at keyword `data_list`  with the path of your target training set.

     ```shell
     python examples/seg/deeplabv3/preprocess/build_seg_data.py \
     		--data_root=[root path of training data] \
     		--data_list=[path of data list file prepared above] \
     		--dst_path=[path to save mindrecords] \
     		--num_shards=8
     ```

   * Note: the training steps use datasets in mindrecord format, while the evaluation steps directly use the data list files.

4. Backbone: download pre-trained backbone from MindCV, here we use [ResNet101](https://download.mindspore.cn/toolkits/mindcv/resnet/resnet101-689c5e77.ckpt).

### Train

Specify `deeplabv3`  or  `deeplabv3plus` at the key word `model` in the config file.

It is highly recommended to use **distributed training** for this DeepLabV3 and DeepLabV3+ implementation.

For distributed training using **`msrun`**, simply run
```shell
msrun --bind_core=True --worker_num [# of devices] python examples/seg/deeplabv3/train.py --config [the path to the config file]
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
bash ascend8p.sh
```

For single-device training, simply set the keyword ``distributed`` to ``False`` in the config file and run:
```shell
python examples/seg/deeplabv3/train.py --config [the path to the config file]
```

**Take msrun command as an example, the training steps are as follow**:

- Step 1: Employ output_stride=16 and fine-tune pretrained resnet101 on *trainaug* dataset. In config file, please specify the path of pretrained backbone checkpoint in keyword `backbone_ckpt_path` and set `output_stride` to `16`.

  ```shell
  # for deeplabv3
  msrun --bind_core=True --worker_num 8 python examples/seg/deeplabv3/train.py --config examples/seg/deeplabv3/config/deeplabv3_s16_dilated_resnet101.yaml

  # for deeplabv3+
  msrun --bind_core=True --worker_num 8 python examples/seg/deeplabv3/train.py --config examples/seg/deeplabv3/config/deeplabv3plus_s16_dilated_resnet101.yaml
  ```

- Step 2: Employ output_stride=8, fine-tune model from step 1 on  *trainaug* dataset with smaller base learning rate. In config file, please specify the path of checkpoint from previous step in `ckpt_path`, set  `ckpt_pre_trained` to `True` and set `output_stride` to `8` .

  ```shell
  # for deeplabv3
  msrun --bind_core=True --worker_num 8 python examples/seg/deeplabv3/train.py --config examples/seg/deeplabv3/config/deeplabv3_s8_dilated_resnet101.yaml

  # for deeplabv3+
  msrun --bind_core=True --worker_num 8 python examples/seg/deeplabv3/train.py --config examples/seg/deeplabv3/config/deeplabv3plus_s8_dilated_resnet101.yaml
  ```
> If use Ascend 910 devices, need to open SATURATION_MODE via `export MS_ASCEND_CHECK_OVERFLOW_MODE="SATURATION_MODE"`.

### Test

For testing the trained model, first specify the path to the model checkpoint at keyword `ckpt_path` in the config file. You could modify `output_stride`, `flip`, `scales` in the config file during inference.

For example, after replacing  `ckpt_path` in config file with [checkpoint](https://download.mindspore.cn/toolkits/mindcv/deeplabv3/deeplabv3_s8_resnet101-a297e7af.ckpt) from 2-step training of deeplabv3, commands below employ os=8 without left-right filpped or muticale inputs.
```shell
python examples/seg/deeplabv3/eval.py --config examples/seg/deeplabv3/config/deeplabv3_s8_dilated_resnet101.yaml
```

## Performance
Experiments are tested on ascend 910 with mindspore 2.3.1 graph mode.



| model name        | params(M) | cards | batch size | jit level | graph compile | ms/step | img/s  | mIoU                | recipe                                                                                                                           | weight      |
| ----------------- | --------- | ----- | ---------- | --------- | ------------- | ------- | ------ | ------------------- | -------------------------------------------------------------------------------------------------------------------------------- | ----------- |
| deeplabv3_s16     | 58.15     | 8     | 32         | O2        | 122s          | 267.91  | 955.54 | 77.33               | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/examples/seg/deeplabv3/config/deeplabv3_s16_dilated_resnet101.yaml)     | [weights](https://download-mindspore.osinfra.cn/toolkits/mindcv/deeplabv3/deeplabv3-s16-best.ckpt) |
| deeplabv3_s8      | 58.15     | 8     | 16         | O2        | 180s          | 390.81  | 327.52 | 79.16\|79.93\|80.14 | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/examples/seg/deeplabv3/config/deeplabv3_s8_dilated_resnet101.yaml)      | [weights](https://download-mindspore.osinfra.cn/toolkits/mindcv/deeplabv3/deeplabv3-s8-best.ckpt) |
| deeplabv3plus_s16 | 59.45     | 8     | 32         | O2        | 207s          | 312.15  | 820.12 | 78.99               | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/examples/seg/deeplabv3/config/deeplabv3plus_s16_dilated_resnet101.yaml) | [weights](https://download-mindspore.osinfra.cn/toolkits/mindcv/deeplabv3/deeplabv3plus-s16-best.ckpt) |
| deeplabv3plus_s8  | 59.45     | 8     | 16         | O2        | 170s          | 403.43  | 217.28 | 80.31\|80.99\|81.10 | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/examples/seg/deeplabv3/config/deeplabv3plus_s8_dilated_resnet101.yaml)  | [weights](https://download-mindspore.osinfra.cn/toolkits/mindcv/deeplabv3/deeplabv3plus-s8-best.ckpt) |



Experiments are tested on ascend 910* with mindspore 2.3.1 graph mode.

*coming soon*

### Notes
- mIoU: mIoU of model "deeplabv3_s8" and "deeplabv3plus_s8" contains 3 results which tested respectively under conditions of no enhance/with MS/with MS and FLIP.
- MS: multiscale inputs during test.
- Flip: adding left-right flipped inputs during test.

As illustrated in [<a href="#references">1</a>], adding left-right flipped inputs or muilt-scale inputs during test could improve the performence. Also, once the model is finally trained, employed output_stride=8 during inference bring improvement over using  output_stride=16.


## References
[1] Chen L C, Papandreou G, Schroff F, et al. Rethinking atrous convolution for semantic image segmentation[J]. arXiv preprint arXiv:1706.05587, 2017.

[2] Chen, Liang-Chieh, et al. "Encoder-decoder with atrous separable convolution for semantic image segmentation." *Proceedings of the European conference on computer vision (ECCV)*. 2018.
