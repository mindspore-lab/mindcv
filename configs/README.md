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

| model name  | params(M) | cards | batch size | resolution | jit level | graph compile | ms/step | img/s   | acc@top1 | acc@top5 | recipe                                                                                              | weight                                                                                                    |
| ----------- | --------- | ----- | ---------- | ---------- | --------- | ------------- | ------- | ------- | -------- | -------- | --------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| densenet121 | 8.06      | 8     | 32         | 224x224    | O2        | 300s          | 47,34   | 5446.81 | 75.67    | 92.77    | [yaml](https://github.com/mindspore-lab/mindcv/blob/main/configs/densenet/densenet_121_ascend.yaml) | [weights](https://download-mindspore.osinfra.cn/toolkits/mindcv/densenet/densenet121-bf4ab27f-910v2.ckpt) |

</div>

Illustration:
- Model: model name in lower case with _ seperator.
- Top-1 and Top-5: Accuracy reported on the validatoin set of ImageNet-1K. Keep 2 digits after the decimal point.
- Params (M): # of model parameters in millions (10^6). Keep **2 digits** after the decimal point
- Batch Size: Training batch size
- Cards: # of cards
- Ms/step: Time used on training per step in ms
- Jit_level: Jit level of mindspore context, which contains 3 levels: O0/O1/O2
- Recipe: Training recipe/configuration linked to a yaml config file.
- Download: url of the pretrained model weights

### Model Checkpoint Format
 The checkpoint (i.e., model weight) name should follow this format:  **{model_name}_{specification}-{sha256sum}.ckpt**, e.g., `poolformer_s12-5be5c4e4.ckpt`.

 You can run the following command and take the first 8 characters of the computing result as the sha256sum value in the checkpoint name.

 ```shell
 sha256sum your_model.ckpt
 ```


#### Training Script Format

For consistency, it is recommended to provide distributed training commands based on `msrun --bind_core=True --worker_num {num_devices} python train.py`, instead of using shell script such as `distrubuted_train.sh`.

  ```shell
  # standalone training on single NPU device
  python train.py --config configs/densenet/densenet_121_gpu.yaml --data_dir /path/to/dataset --distribute False

  # distributed training on NPU divices
  msrun --bind_core=True --worker_num 8 python train.py --config configs/densenet/densenet_121_ascend.yaml --data_dir /path/to/imagenet

  ```

#### URL and Hyperlink Format
Please use **absolute path** in the hyperlink or url for linking the target resource in the readme file and table.
