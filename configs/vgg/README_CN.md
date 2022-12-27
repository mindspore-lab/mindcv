# VGGNet
> 想了解更多细节，请参考： [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

## 模型简介

<div align=center>

![](vggnet.png)
</div>

VGGNet是图像分类任务的一个里程碑。它首次采用3x3的卷积核而不是5x5或者7x7的卷积核，并且它证明了5x5或者7x7的卷积核可以被3x3的卷积核所代替并且实现相同的效果。更重要的是，VGGNet首次将模型扩展到16-19层。相比于GoogleLeNet和AlexNet，VGGNet可以在 [ImageNet-1K dataset](https://www.image-net.org/download.php)上实现更好的模型表现。

## 性能指标

| Model           | Context   |  Top-1 (%)  | Top-5 (%)  |  Params (M)    | Train T. | Infer T. |  Download | Config | Log |
|-----------------|-----------|-------|-------|------------|-------|--------|---|--------|--------------|
| VGG-16  | D910x8-G | -     | -     | -       | -s/epoch | -ms/step | [model]() | [cfg]() | [log]() |
| VGG-19 | D910x8-G | -     | -     | -       | -s/epoch | -ms/step | [model]() | [cfg]() | [log]() | 


#### 备注

- 所有的模型均在ImageNet-1K上进行训练，并且top-1 accuracy和top-5 accuracy被报告。
- Context: GPU_TYPE x pieces - G/F, G - graph mode, F - pynative mode with ms function.  

## 快速开始

<details>
<summary>准备</summary>

#### 安装
请参考mindcv的[安装指示](https://github.com/mindspore-ecosystem/mindcv#installation) in MindCV.

#### 数据集准备
请下载[ImageNet-1K](htps://www.image-net.org/download.php)数据集用于训练和验证。
</details>

<details>
<summary>训练</summary>

- **超参数.** 可复现训练结果的配置设置存放在 `mindcv/configs/vgg`文件夹。例如，为了按照某个配置进行训练，你可以运行:

  ```shell
  # train VGG-19 on 8 GPUs
  mpirun -n 8 python train.py --config path/to/vgg/yaml/file --data_dir /path/to/imagenet
  ```

  注意GPU或者昇腾芯片的数量以及batch size都会影响复现结果。为了最大程度的复现结果，推荐采用相同显卡数量和相同batch size进行训练。

详细的参数可以参考[config.py](../../config.py)。
</details>

<details>
<summary>验证</summary>

- 为了验证模型，你可以使用`validate.py`。 这里有一个例子来验证VGG-19模型的精准度。

  ```shell
  python validate.py --config path/to/vgg/yaml/file --data_dir /path/to/imagenet --ckpt_path /path/to/vgg/file.ckpt
  ```

</details>


<details>
<summary>部署（可选）</summary>

请参考mindcv中的部署教程。 
</details>


