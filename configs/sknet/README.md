# SKNet

***

> [SKNet: Selective Kernel Networks](https://arxiv.org/pdf/1903.06586.pdf)

## Introduction

***

The core idea of SKNet: SK Convolution

1. Split is to perform different receptive field convolution operations. The upper branch is a 3 x 3 kernel with dilate
   size=1, and the lower one is a 3 x 3 convolution with dilate size=2.
2. Fuse performs feature fusion, superimposes the convolutional features of the two branches, and then performs the
   standard SE process.
3. Select is to aggregate the feature maps of differently sized kernels according to the selection weights.

## Results
Coming Soon
