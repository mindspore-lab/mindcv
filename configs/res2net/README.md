# Res2Net

***

> [Res2Net: A New Multi-scale Backbone Architecture](https://arxiv.org/pdf/1904.01169.pdf)

## Introduction

***
We propose a novel building block for CNNs, namely Res2Net, by constructing hierarchical residual-like connections
within one single residual block. The Res2Net represents multi-scale features at a granular level and increases the
range of receptive fields for each network layer. The proposed Res2Net block can be plugged into the state-of-the-art
backbone CNN models, e.g. , ResNet, ResNeXt, BigLittleNet, and DLA. We evaluate the Res2Net block on all these models
and demonstrate consistent performance gains over baseline models.
![](res2net.png)

## Benchmark

***

| Model          | Context  | Top-1 (%) | Top-5 (%) | Params (M) | Train T.   | Infer T.   | Download  | Config  | Log     |
| -------------- | -------- | --------- | --------- | ---------- | ---------- | ---------- | --------- | ------- | ------- |
| Res2Net50      | D910x8-G | 79.35     | 94.64     | 25.76      | 246s/epoch | 28.5ms/step | [model]() | [cfg]() | [log]() |
| Res2Net101     | D910x8-G | 79.56     | 94.70     | 45.33      | 467s/epoch | 46.0ms/step | [model]() | [cfg]() | [log]() |
| Res2Net50      | D910x8-G | 80.32     | 95.09     | 25.77      | 250s/epoch | 29.6ms/step | [model]() | [cfg]() | [log]() |
| Res2Net101-v1b | D910x8-G | 81.26     | 95.41     | 45.35      | 435s/epoch | 42.4ms/step | [model]() | [cfg]() | [log]() |

#### Notes

- All models are trained on ImageNet-1K training set and the top-1 accuracy is reported on the validatoin set.
- Context: GPU_TYPE x pieces - G/F, G - graph mode, F - pynative mode with ms function.  

## Examples

***

### Train

- The [yaml config files](../../configs) that yield competitive results on ImageNet for different models are listed in
  the `configs` folder. To trigger training using preset yaml config.

  ```shell
  mpirun -n 8 python train.py --config configs/res2net/res2net_50_gpu.yaml --data_dir /path/to/imagenet
  ```

Detailed adjustable parameters and their default value can be seen in [config.py](../../config.py).

### Eval

- To validate the model, you can use `validate.py`. Here is an example for res2net50 to verify the accuracy of your
  training.

  ```shell
  python validate.py --config configs/res2net/res2net_50_gpu.yaml --data_dir /path/to/imagenet --ckpt_path /path/to/res2net50.ckpt
  ```
