# MobileNetV3
> [Searching for MobileNetV3](https://arxiv.org/pdf/1905.02244v5.pdf)

## Introduction
***

MobileNet v3 was published in 2019, and this v3 version combines the deep separable convolution of v1, the Inverted Residuals and Linear Bottleneck of v2, and the SE module to search the configuration and parameters of the network using NAS (Neural Architecture Search).MobileNetV3 first uses MnasNet to perform a coarse structure search, and then uses reinforcement learning to select the optimal configuration from a set of discrete choices. Afterwards, MobileNetV3 then fine-tunes the architecture using NetAdapt, which exemplifies NetAdapt's complementary capability to tune underutilized activation channels with a small drop.

mobilenet-v3 offers two versions, mobilenet-v3 large and mobilenet-v3 small, for situations with different resource requirements. The paper mentions that mobilenet-v3 small, for the imagenet classification task, has an accuracy The paper mentions that mobilenet-v3 small achieves about 3.2% better accuracy and 15% less time than mobilenet-v2 for the imagenet classification task, mobilenet-v3 large achieves about 4.6% better accuracy and 5% less time than mobilenet-v2 for the imagenet classification task, mobilenet-v3 large achieves the same accuracy and 25% faster speedup in COCO compared to v2 The improvement in the segmentation algorithm is also observed.

![](./MobileNetV3_Block.png)

## Results
model is under testing, comming soon



