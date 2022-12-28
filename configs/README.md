### File Structure and Naming
This folder contains training recipes and model readme files for each model. The folder structure and naming rule of model configurations are as follows.


```
    ├── configs
        ├── model_a  // model name in lowcase with _ seperator
        │   ├─ model_a_small_ascend.yaml // training recipe denated as {model_name}_{specifiation}_{hardware}.yaml
        |   ├─ model_a_large_gpu.yaml
        │   ├─ README.md   //readme file containing performance results and pretrained weight urls
        │   └─ README_CN.md  //readme file in Chinese
        ├── model_b 
        │   ├─ model_b_32_ascend.yaml 
        |   ├─ model_l_16_ascend.yaml
        │   ├─ README.md   
        │   └─ README_CN.md 
        ├── README.md //this file
```

### Model Readme Writing Guideline
The model readme file in each sub-folder provides the introduction, reproduced results, and running guideline for each model. 

Please follow the outline structure and **table format** shown in [densenet/README.md](densenet/README.md) when contributing your models :) 

- Table Format

| Model           | Context   |  Top-1 (%) | Top-5 (%)  |  Params (M) | Recipe  | Download |
|-----------------|-----------|------------|------------|-------------|---------|----------|
| DenseNet121 | D910x8-G | 75.64     | 92.84     | 8.06       | [YAML]() | [weights]() \| [log]()  |

- Training Script Format

For consistency, it is recommended to provide distributed training commands based on `mpirun -n {num_devices} python train.py`, instead of using shell script such as `distrubuted_train.sh`. 

  ```shell
  # standalone training
  python train.py --config configs/densenet/densenet_121_gpu.yaml --data_dir /path/to/dataset
  
  # distributed training
  mpirun -n 8 python train.py --config configs/densenet/densenet_121_gpu.yaml --data_dir /path/to/imagenet
  
  ```
  
  


