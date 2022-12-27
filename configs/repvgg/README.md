# RepVGG

***
> [RepVGG: Making VGG-style ConvNets Great Again](https://arxiv.org/pdf/2101.03697.pdf)

## Introduction

***
RepVGG, a vgg-style architecture that outperforms many complex models

Its main highlights are:

1) The model has a normal (a.k.a. feedforward) structure like vgg, without any other branches, each layer takes the
   output of its only previous layer as input, and feeds the output to its only next layer.

2) The body of the model uses only 3 × 3 conv and ReLU.

3) The specific architecture (including specific depth and layer width) is instantiated without automatic search, manual
   refinement, compound scaling, and other complicated designs.

## Results
model is under testing, comming soon
  ```
