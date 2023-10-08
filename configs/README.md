### File Structure and Naming
This folder contains training recipes and model readme files for each model. The folder structure and naming rule of model configurations are as follows.


```
    ├── configs
        ├── model_a                         // model name in lower case with _ seperator
        │   ├─ model_a_small_ascend.yaml    // training recipe denated as {model_name}_{specification}_{hardware}.yaml
        |   ├─ model_a_large_gpu.yaml
        │   ├─ README.md                    //readme file containing performance results and pretrained weight urls
        │   └─ README_CN.md                 //readme file in Chinese
        ├── model_b
        │   ├─ model_b_32_ascend.yaml
        |   ├─ model_l_16_ascend.yaml
        │   ├─ README.md
        │   └─ README_CN.md
        ├── README.md //this file
```

> Note: Our training recipes are verified on specific hardware, and the suffix `hardware` (ascend or gpu) in the
> file name of training recipes indicates different hardware. Since Mindspore operators have different precision and
> performance on different hardware, different training recipes are required under different hardware. However, if you
> want to train on another hardware (e.g. GPU) using the training recipe under specific hardware (e.g. Ascend), you only
> need to make minor or no adjustments to the hyperparameters, because the training recipe has a certain degree of
> generalization across different hardware.

### Model Readme Writing Guideline
The model readme file in each sub-folder provides the introduction, reproduced results, and running guideline for each model.

Please follow the outline structure and **table format** shown in [densenet/README.md](https://github.com/mindspore-lab/mindcv/blob/main/configs/densenet/README.md) when contributing your models :)

#### Table Format

<div align="center">

| Model        | Context  | Top-1 (%) | Top-5 (%) | Params (M) | Recipe                                                                                              | Download                                                                                           |
|--------------|----------|-----------|-----------|------------|-----------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|
| densenet_121 | D910x8-G | 75.64     | 92.84     | 8.06       | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/densenet/densenet_121_ascend.yaml) | [weights](https://download.mindspore.cn/toolkits/mindcv/densenet/densenet121-120_5004_Ascend.ckpt) |

</div>

Illustration:
- Model: model name in lower case with _ seperator.
- Context: Training context denoted as {device}x{pieces}-{MS mode}, where mindspore mode can be G - graph mode or F - pynative mode with ms function. For example, D910x8-G is for training on 8 pieces of Ascend 910 NPU using graph mode.
- Top-1 and Top-5: Accuracy reported on the validatoin set of ImageNet-1K. Keep 2 digits after the decimal point.
- Params (M): # of model parameters in millions (10^6). Keep **2 digits** after the decimal point
- Recipe: Training recipe/configuration linked to a yaml config file.
- Download: url of the pretrained model weights

### Model Checkpoint Format
 The checkpoint (i.e., model weight) name should follow this format:  **{model_name}_{specification}-{sha256sum}.ckpt**, e.g., `poolformer_s12-5be5c4e4.ckpt`.

 You can run the following command and take the first 8 characters of the computing result as the sha256sum value in the checkpoint name.

 ```shell
 sha256sum your_model.ckpt
 ```


#### Training Script Format

For consistency, it is recommended to provide distributed training commands based on `mpirun -n {num_devices} python train.py`, instead of using shell script such as `distrubuted_train.sh`.

  ```shell
  # standalone training on a gpu or ascend device
  python train.py --config configs/densenet/densenet_121_gpu.yaml --data_dir /path/to/dataset --distribute False

  # distributed training on gpu or ascend divices
  mpirun -n 8 python train.py --config configs/densenet/densenet_121_ascend.yaml --data_dir /path/to/imagenet

  ```
  > If the script is executed by the root user, the `--allow-run-as-root` parameter must be added to `mpirun`.

#### URL and Hyperlink Format
Please use **absolute path** in the hyperlink or url for linking the target resource in the readme file and table.
