# RegNet
> [Designing Network Design Spaces](https://arxiv.org/pdf/2003.13678.pdf)

## Introduction

In this work, we present a new network design paradigm that combines the advantages of manual design and NAS. Instead of focusing on designing individual network instances, we design design spaces that parametrize populations of networks. Like in manual design, we aim for interpretability and to discover general design principles that describe networks that are simple, work well, and generalize across settings. Like in NAS, we aim to take advantage of semi-automated procedures to help achieve these goals The general strategy we adopt is to progressively design simplified versions of an initial, relatively unconstrained, design space while maintaining or improving its quality. The overall process is analogous to manual design, elevated to the population level and guided via distribution estimates of network design spaces. As a testbed for this paradigm, our focus is on exploring network structure (e.g., width, depth, groups, etc.) assuming standard model families including VGG, ResNet, and ResNeXt. We start with a relatively unconstrained design space we call AnyNet (e.g., widths and depths vary freely across stages) and apply our humanin-the-loop methodology to arrive at a low-dimensional design space consisting of simple “regular” networks, that we call RegNet. The core of the RegNet design space is simple: stage widths and depths are determined by a quantized linear function. Compared to AnyNet, the RegNet design space has simpler models, is easier to interpret, and has a higher concentration of good models.

![](regnet.png)

## Results
model is under testing, comming soon
