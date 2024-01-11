This folder contains scripts for fine-tuning on your own custom dataset with a pretrained model offered by MindCV, do refer to [tutorials](https://mindspore-lab.github.io/mindcv/how_to_guides/finetune_with_a_custom_dataset/) for details.

### split_files.py
```shell
python examples/finetune/split_files.py
```
This file taking Aircraft dataset as an example, shows how to manually reorganize data into a tree-structure directory according to an annotation file. Note that it's only for Aircraft dataset but not a general one, you'd better check the content before running it.

### read_images_online.py

This is an example demonstrating how to read the raw images as well as the labels into a `GeneratorDataset` object, which is a `MappableDataset` class that can be read directly by models. You are recommended to insert this part into your data preprocessing script or training script, but not run it alone.

### finetune.py
```shell
python examples/finetune/finetune.py --config=./configs/mobilenetv3/mobilnet_v3_small_ascend.yaml
```
A script for fine-tuning with some example code of fine-tuning methods in it (all settings during fine-tuning are inside the config file, for more details, please refer to the tutorial mentioned above).
