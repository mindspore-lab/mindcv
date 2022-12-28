### File Structure
This folder contains training recipes and model readme files for each model. The folder structure and naming rule of model configurations are as follows.


      ```text
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

### Model Readme (for contributors)
The model readme file in each sub-folder provides the introduction, reproduced results, and running guideline for each model. 

Please follow the outline structure and table format shown in [densenet/README.md](configs/densenet/README.md) for contributing your models :)

