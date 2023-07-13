# Release Note

## 0.2.2 (2023/6/16)

1. New version `0.2.2` is released! We upgrade to support `MindSpore` v2.0 while maintaining compatibility of v1.8
2. New models:
   - [ConvNextV2](configs/convnextv2)
   - mini of [CoAT](configs/coat)
   - 1.3 of [MnasNet](configs/mnasnet)
   - AMP(O3) version of [ShuffleNetV2](configs/shufflenetv2)
3. New features:
   - Gradient Accumulation
   - DynamicLossScale for customized [TrainStep](mindcv/utils/train_step.py)
   - OneCycleLR and CyclicLR learning rate scheduler
   - Refactored Logging
   - Pyramid Feature Extraction
4. Bug fixes:
   - Serving Deployment Tutorial(mobilenet_v3 doesn't work on ms1.8 when using Ascend backend)
   - Some broken links on our documentation website.

## 0.2.1

- 2023/6/2
1. New version: `0.2.1` is released!
2. New [documents](https://mindspore-lab.github.io/mindcv/) is online!

- 2023/5/30
1. New Models:
    - AMP(O2) version of [VGG](configs/vgg)
    - [GhostNet](configs/ghostnet)
    - AMP(O3) version of [MobileNetV2](configs/mobilenetv2) and [MobileNetV3](configs/mobilenetv3)
    - (x,y)_(200,400,600,800)mf of [RegNet](configs/regnet)
    - b1g2, b1g4 & b2g4 of [RepVGG](configs/repvgg)
    - 0.5 of [MnasNet](configs/mnasnet)
    - b3 & b4 of [PVTv2](configs/pvtv2)
2. New Features:
    - 3-Augment, Augmix, TrivialAugmentWide
3. Bug Fixes:
    - ViT pooling mode

- 2023/04/28
1. Add some new models, listed as following
    - [VGG](configs/vgg)
    - [DPN](configs/dpn)
    - [ResNet v2](configs/resnetv2)
    - [MnasNet](configs/mnasnet)
    - [MixNet](configs/mixnet)
    - [RepVGG](configs/repvgg)
    - [ConvNeXt](configs/convnext)
    - [Swin Transformer](configs/swintransformer)
    - [EdgeNeXt](configs/edgenext)
    - [CrossViT](configs/crossvit)
    - [XCiT](configs/xcit)
    - [CoAT](configs/coat)
    - [PiT](configs/pit)
    - [PVT v2](configs/pvtv2)
    - [MobileViT](configs/mobilevit)
2. Bug fix:
    - Setting the same random seed for each rank
    - Checking if options from yaml config exist in argument parser
    - Initializing flag variable as `Tensor` in Optimizer `Adan`

## 0.2.0

- 2023/03/25
1. Update checkpoints for pretrained ResNet for better accuracy
    - ResNet18 (from 70.09 to 70.31 @Top1 accuracy)
    - ResNet34 (from 73.69 to 74.15 @Top1 accuracy)
    - ResNet50 (from 76.64 to 76.69 @Top1 accuracy)
    - ResNet101 (from 77.63 to 78.24 @Top1 accuracy)
    - ResNet152 (from 78.63 to 78.72 @Top1 accuracy)
2. Rename checkpoint file name to follow naming rule ({model_scale-sha256sum.ckpt}) and update download URLs.

- 2023/03/05
1. Add Lion (EvoLved Sign Momentum) optimizer from paper https://arxiv.org/abs/2302.06675
    - To replace adamw with lion, LR is usually 3-10x smaller, and weight decay is usually 3-10x larger than adamw.
2. Add 6 new models with training recipes and pretrained weights for
    - [HRNet](configs/hrnet)
    - [SENet](configs/senet)
    - [GoogLeNet](configs/googlenet)
    - [Inception V3](configs/inceptionv3)
    - [Inception V4](configs/inceptionv4)
    - [Xception](configs/xception)
3. Support gradient clip
4. Arg name `use_ema` changed to **`ema`**, add `ema: True` in yaml to enable EMA.

## 0.1.1

- 2023/01/10
1. MindCV v0.1 released! It can be installed via PyPI `pip install mindcv` now.
2. Add training recipe and trained weights of googlenet, inception_v3, inception_v4, xception

## 0.1.0

- 2022/12/09
1. Support lr warmup for all lr scheduling algorithms besides cosine decay.
2. Add repeated augmentation, which can be enabled by setting `--aug_repeats` to be a value larger than 1 (typically, 3 or 4 is a common choice).
3. Add EMA.
4. Improve BCE loss to support mixup/cutmix.

- 2022/11/21
1. Add visualization for loss and acc curves
2. Support epochwise lr warmup cosine decay (previous is stepwise)

- 2022/11/09
1. Add 7 pretrained ViT models.
2. Add RandAugment augmentation.
3. Fix CutMix efficiency issue and CutMix and Mixup can be used together.
4. Fix lr plot and scheduling bug.

- 2022/10/12
1. Both BCE and CE loss now support class-weight config, label smoothing, and auxiliary logit input (for networks like inception).

## 0.0.1-beta

- 2022/09/13
1. Add Adan optimizer (experimental)

## MindSpore Computer Vision 0.0.1

### Models

`mindcv.models` now expose `num_classes` and `in_channels` as constructor arguments:

- Add DenseNet models and pre-trained weights
- Add GoogLeNet models and pre-trained weights
- Add Inception V3 models and pre-trained weights
- Add Inception V4 models and pre-trained weights
- Add MnasNet models and pre-trained weights
- Add MobileNet V1 models and pre-trained weights
- Add MobileNet V2 models and pre-trained weights
- Add MobileNet V3 models and pre-trained weights
- Add ResNet models and pre-trained weights
- Add ShuffleNet V1 models and pre-trained weights
- Add ShuffleNet V2 models and pre-trained weights
- Add SqueezeNet models and pre-trained weights
- Add VGG models and pre-trained weights
- Add ViT models and pre-trained weights

### Dataset

`mindcv.data` now expose:

- Add Mnist dataset
- Add FashionMnist dataset
- Add Imagenet dataset
- Add CIFAR10 dataset
- Add CIFAR100 dataset

### Loss

`mindcv.loss` now expose:

- Add BCELoss
- Add CrossEntropyLoss

### Optimizer

`mindcv.optim` now expose:

- Add SGD optimizer
- Add Momentum optimizer
- Add Adam optimizer
- Add AdamWeightDecay optimizer
- Add RMSProp optimizer
- Add Adagrad optimizer
- Add Lamb optimizer

### Learning_Rate Scheduler

`mindcv.scheduler` now expose:

- Add WarmupCosineDecay learning rate scheduler
- Add ExponentialDecayLR learning rate scheduler
- Add Constant learning rate scheduler

### Release

mindcv-0.0.1.apk

mindcv-0.0.1-py3-none-any.whl.sha256

mindcv-0.0.1-py3-none-any.whl
