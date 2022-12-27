# InceptionV3

***

> [InceptionV3: Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/pdf/1512.00567.pdf)

## Introduction

***
InceptionV3 is an upgraded version of GoogleNet. One of the most important improvements of V3 is Factorization, which
decomposes 7x7 into two one-dimensional convolutions (1x7, 7x1), and 3x3 is the same (1x3, 3x1), such benefits, both It
can accelerate the calculation (excess computing power can be used to deepen the network), and can split 1 conv into 2
convs, which further increases the network depth and increases the nonlinearity of the network. It is also worth noting
that the network input from 224x224 has become 299x299, and 35x35/17x17/8x8 modules are designed more precisely. In
addition, V3 also adds batch normalization, which makes the model converge more quickly, which plays a role in partial
regularization and effectively reduces overfitting.
![](InceptionV3网络.jpg)

## Results
model is under testing, comming soon
