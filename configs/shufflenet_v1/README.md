# ShuffleNetV1
> [ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](https://arxiv.org/pdf/1707.01083.pdf)

## Introduction
***

ShuffleNet is a computationally efficient CNN model proposed by KuangShi Technology in 2017, which, like MobileNet and SqueezeNet, etc., is mainly intended to be applied in mobile. Therefore, the design goal of ShuffleNet is also how to use limited computational resources to achieve the best model accuracy, which requires a good balance between speed and accuracy.ShuffleNet uses two operations at its core: pointwise group convolution and channel shuffle, which greatly reduces the model computation while maintaining accuracy. This greatly reduces the computational effort of the model while maintaining accuracy. The main design ideas of CNN models for mobile are mainly two: model structure design and model compression, ShuffleNet and MobileNet belong to the former, both of which design more efficient network structures to achieve smaller and faster models, instead of compressing or migrating a large trained model.

![](./ShuffleNetV1_Block.png)

## Results
model is under testing, comming soon