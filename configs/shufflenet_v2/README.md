# ShuffleNetV2
> [ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://arxiv.org/pdf/1807.11164.pdf)

## Introduction
***

A key point was raised in ShuffleNetV2, where previous lightweight networks were guided by computing an indirect measure of network complexity, namely FLOPs. The speed of lightweight networks is described by calculating the amount of floating point operations. But the speed of operation was never considered directly. The running speed in mobile devices needs to consider not only FLOPs, but also other factors such as memory accesscost and platform characterics.

Therefore, based on these two principles, ShuffleNetV2 proposes four effective network design principles.

- MAC is minimized when the input feature matrix of the convolutional layer is equal to the output feature matrixchannel (when FLOPs are kept constant).
- MAC increases when the groups of GConv increase (while keeping FLOPs constant).
- the higher the fragmentation of the network design, the slower the speed.
- The impact of Element-wise operation is not negligible.

![](./ShuffleNetV2_Block.png)


## Results
model is under testing, comming soon

