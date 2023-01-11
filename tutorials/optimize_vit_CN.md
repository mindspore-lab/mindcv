# Vision Transformer图像分类优化

[![下载Notebook](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.8/resource/_static/logo_notebook.png)](https://download.mindspore.cn/toolkits/mindcv/tutorials/optimize_vit_CN.ipynb)
&emsp;

在此教程中，您将学会如何使用MindCV套件来对vit的图片分类任务的表现进行优化，以解决在自定义数据集上使用vit模型进行图像分类的问题。在深度学习任务中，我们常常遇到训练数据不足的问题，此时直接训练整个网络往往难以达到理想的精度。一个比较好的做法是，使用一个在大规模数据集上(与任务数据较为接近)预训练好的模型，然后使用该模型来初始化网络的权重参数或作为固定特征提取器应用于特定的任务中。

此教程将以使用ImageNet上预训练的vit_b_32_224规格的vit模型为例，介绍通过整体网络微调优化vit模型的策略，解决小样本情况下狼和狗的图像分类问题。

## Vision Transformer（ViT）简介

近些年，随着基于自注意（Self-Attention）结构的模型的发展，特别是Transformer模型的提出，极大地促进了自然语言处理模型的发展。由于Transformers的计算效率和可扩展性，它已经能够训练具有超过100B参数的空前规模的模型。

ViT则是自然语言处理和计算机视觉两个领域的融合结晶。在不依赖卷积操作的情况下，依然可以在图像分类任务上达到很好的效果。

### 模型结构

ViT模型的主体结构是基于Transformer模型的Encoder部分（部分结构顺序有调整，如：Normalization的位置与标准Transformer不同），其结构图[1]如下：

![vit-architecture](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/tutorials/application/source_zh_cn/cv/images/vit_architecture.png)
<center> 图[1] </center>

### 模型特点

ViT模型主要应用于图像分类领域。因此，其模型结构相较于传统的Transformer有以下几个特点：

1. 数据集的原图像被划分为多个patch后，将二维patch（不考虑channel）转换为一维向量，再加上类别向量与位置向量作为模型输入。
2. 模型主体的Block结构是基于Transformer的Encoder结构，但是调整了Normalization的位置，其中，最主要的结构依然是Multi-head Attention结构。
3. 模型在Blocks堆叠后接全连接层，接受类别向量的输出作为输入并用于分类。通常情况下，我们将最后的全连接层称为Head，Transformer Encoder部分为backbone。

下面将通过代码实例来详细解释基于ViT实现狼和狗的图像分类任务。

## 环境准备与数据读取

开始实验之前，请确保本地已经安装了Python环境并正确安装了[MindSPore](https://mindspore.cn/install)和[MindCV](https://github.com/mindspore-lab/mindcv/blob/main/README.md#Installation)。

## 数据加载与处理

首先导入相关模块，配置相关超参数并读取数据集，该部分代码在MindCV套件中都有API可直接调用。

首先我们需要下载本案例的数据集，运行第一段代码时会自动下载并解压，请确保你的数据集路径如以下结构。


```text
data/
└── Canidae
    ├── train
    │   ├── dogs
    │   └── wolves
    └── val
        ├── dogs
        └── wolves
```

```python
import sys
sys.path.append('../')

from mindcv.utils.download import DownLoad
import os

dataset_url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/intermediate/Canidae_data.zip"
root_dir = "./"

if not os.path.exists(os.path.join(root_dir, 'data/Canidae')):
    DownLoad().download_and_extract_archive(dataset_url, root_dir)
```

```text
11882496B [00:00, 22486394.82B/s]
```

通过调用`mindcv.data`中的`create_dataset`函数，我们可轻松地加载预设的数据集以及自定义的数据集。
- 当参数`name`设为空时，指定为自定义数据集。(默认值)
- 当参数`name`设为`MNIST`, `CIFAR10`等标准数据集名称时，指定为预设数据集。

同时，我们需要设定数据集的路经`data_dir`和数据切分的名称`split` (如train, val)，以加载对应的训练集或者验证集。


```python
from mindcv.data import create_dataset, create_transforms, create_loader

num_workers = 8

# 数据集目录路径
data_dir = "data/Canidae"

# 加载自定义数据集
dataset_train = create_dataset(root=data_dir, split='train', num_parallel_workers=num_workers)
```

首先我们通过调用`create_transforms`函数, 获得预设的数据处理和增强策略(transform list)，此任务中，因狼狗图像和ImageNet数据一致（即domain一致），我们指定参数`dataset_name`为ImageNet，直接用预设好的ImageNet的数据处理和图像增强策略。`create_transforms` 同样支持多种自定义的处理和增强操作，以及自动增强策略(AutoAug)。详见API说明。 

我们将得到的transform list传入`create_loader()`，并指定`batch_size`和其他参数，即可完成训练和验证数据的准备，返回`Dataset` Object，作为模型的输入。


```python
# 定义和获取数据处理及增强操作
trans_train = create_transforms(dataset_name='ImageNet', is_training=True)

loader_train = create_loader(
        dataset=dataset_train,
        batch_size=16,
        is_training=True,
        num_classes=2,
        transform=trans_train,
        num_parallel_workers=num_workers,
    )
```

## 模型解析

下面将通过代码来细致剖析ViT模型的内部结构。

### Transformer基本原理

Transformer模型源于2017年的一篇文章[2]。在这篇文章中提出的基于Attention机制的编码器-解码器型结构在自然语言处理领域获得了巨大的成功。模型结构如下图所示：

![transformer-architecture](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.9/tutorials/application/source_zh_cn/cv/images/transformer_architecture.png)

其主要结构为多个Encoder和Decoder模块所组成，其中Encoder和Decoder的详细结构如下图[2]所示：

![encoder-decoder](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.9/tutorials/application/source_zh_cn/cv/images/encoder_decoder.png)
<center> 图[2] </center>

Encoder与Decoder由许多结构组成，如：多头注意力（Multi-Head Attention）层、Feed Forward层、Normaliztion层和残差连接（Residual Connection，图中的“Add”）。

其中最重要的结构是多头注意力（Multi-Head Attention）结构，该结构基于自注意力（Self-Attention）机制，是多个Self-Attention的并行组成。所以，理解了Self-Attention就抓住了Transformer的核心。

#### Attention模块

以下是Self-Attention的解释，其核心内容是为输入向量的每个单词学习一个权重。通过给定一个任务相关的查询向量Query向量，计算Query和各个Key的相似性或者相关性得到注意力分布，即得到每个Key对应Value的权重系数，然后对Value进行加权求和得到最终的Attention数值。

在Self-Attention中:

1. 最初的输入向量首先会经过Embedding层映射成Q（Query），K（Key），V（Value）三个向量，由于是并行操作，所以代码中是映射成为dim x 3的向量然后进行分割，换言之，如果你的输入向量为一个向量序列（$x_1$，$x_2$，$x_3$），其中的$x_1$，$x_2$，$x_3$都是一维向量，那么每一个一维向量都会经过Embedding层映射出Q，K，V三个向量，只是Embedding矩阵不同，矩阵参数也是通过学习得到的。**这里大家可以认为，Q，K，V三个矩阵是发现向量之间关联信息的一种手段，需要经过学习得到，至于为什么是Q，K，V三个，主要是因为需要两个向量点乘以获得权重，又需要另一个向量来承载权重向加的结果，所以，最少需要3个矩阵。**

$$
\begin{cases}
q_i = W_q \cdot x_i & \\
k_i = W_k \cdot x_i,\hspace{1em} &i = 1,2,3 \ldots \\
v_i = W_v \cdot x_i &
\end{cases}
\tag{1}
$$

![self-attention1](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.9/tutorials/application/source_zh_cn/cv/images/self_attention_1.png)

2. 自注意力机制的自注意主要体现在它的Q，K，V都来源于其自身，也就是该过程是在提取输入的不同顺序的向量的联系与特征，最终通过不同顺序向量之间的联系紧密性（Q与K乘积经过Softmax的结果）来表现出来。**Q，K，V得到后就需要获取向量间权重，需要对Q和K进行点乘并除以维度的平方根，对所有向量的结果进行Softmax处理，通过公式(2)的操作，我们获得了向量之间的关系权重。**

$$
\begin{cases}
a_{1,1} = q_1 \cdot k_1 / \sqrt d \\
a_{1,2} = q_1 \cdot k_2 / \sqrt d \\
a_{1,3} = q_1 \cdot k_3 / \sqrt d
\end{cases}
\tag{2}
$$

![self-attention3](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.9/tutorials/application/source_zh_cn/cv/images/self_attention_3.png)

$$ Softmax: \hat a_{1,i} = exp(a_{1,i}) / \sum_j exp(a_{1,j}),\hspace{1em} j = 1,2,3 \ldots \tag{3}$$

![self-attention2](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.9/tutorials/application/source_zh_cn/cv/images/self_attention_2.png)

3. 其最终输出则是通过V这个映射后的向量与Q，K经过Softmax结果进行weight sum获得，这个过程可以理解为在全局上进行自注意表示。**每一组Q，K，V最后都有一个V输出，这是Self-Attention得到的最终结果，是当前向量在结合了它与其他向量关联权重后得到的结果。**

$$
b_1 = \sum_i \hat a_{1,i}v_i,\hspace{1em} i = 1,2,3...
\tag{4}
$$

通过下图可以整体把握Self-Attention的全部过程。

![self-attention](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.9/tutorials/application/source_zh_cn/cv/images/self_attention_process.png)

多头注意力机制就是将原本self-Attention处理的向量分割为多个Head进行处理，这一点也可以从代码中体现，这也是attention结构可以进行并行加速的一个方面。

总结来说，多头注意力机制在保持参数总量不变的情况下，将同样的query, key和value映射到原来的高维空间（Q,K,V）的不同子空间(Q_0,K_0,V_0)中进行自注意力的计算，最后再合并不同子空间中的注意力信息。

所以，对于同一个输入向量，多个注意力机制可以同时对其进行处理，即利用并行计算加速处理过程，又在处理的时候更充分的分析和利用了向量特征。下图展示了多头注意力机制，其并行能力的主要体现在下图中的$a_1$和$a_2$是同一个向量进行分割获得的。

![multi-head-attention](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.9/tutorials/application/source_zh_cn/cv/images/multi_head_attention.png)

以下是vision套件中的Multi-Head Attention代码，结合上文的解释，代码清晰的展现了这一过程。


```python
from mindspore import nn, ops, Tensor


class Attention(nn.Cell):
    """
    Attention layer implementation, Rearrange Input -> B x N x hidden size.
    Args:
        dim (int): The dimension of input features.
        num_heads (int): The number of attention heads. Default: 8.
        keep_prob (float): The keep rate, greater than 0 and less equal than 1. Default: 1.0.
        attention_keep_prob (float): The keep rate for attention. Default: 1.0.
    Returns:
        Tensor, output tensor.
    Examples:
        >>> ops = Attention(768, 12)
    """

    def __init__(self,
                 dim: int,
                 num_heads: int = 8,
                 keep_prob: float = 1.0,
                 attention_keep_prob: float = 1.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = Tensor(head_dim ** -0.5)

        self.qkv = nn.Dense(dim, dim * 3)
        self.attn_drop = nn.Dropout(attention_keep_prob)
        self.out = nn.Dense(dim, dim)
        self.out_drop = nn.Dropout(keep_prob)

        self.mul = ops.Mul()
        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()
        self.unstack = ops.Unstack(axis=0)
        self.attn_matmul_v = ops.BatchMatMul()
        self.q_matmul_k = ops.BatchMatMul(transpose_b=True)
        self.softmax = nn.Softmax(axis=-1)

    def construct(self, x):
        """Attention construct."""
        b, n, c = x.shape
        qkv = self.qkv(x)
        qkv = self.reshape(qkv, (b, n, 3, self.num_heads, c // self.num_heads))
        qkv = self.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = self.unstack(qkv)

        attn = self.q_matmul_k(q, k)
        attn = self.mul(attn, self.scale)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        out = self.attn_matmul_v(attn, v)
        out = self.transpose(out, (0, 2, 1, 3))
        out = self.reshape(out, (b, n, c))
        out = self.out(out)
        out = self.out_drop(out)

        return out
```

### Transformer Encoder

在了解了Self-Attention结构之后，通过与Feed Forward，Residual Connection等结构的拼接就可以形成Transformer的基础结构，接下来就利用Self-Attention来构建ViT模型中的TransformerEncoder部分，类似于构建了一个Transformer的编码器部分，如下图[1]所示：

![vit-encoder](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.9/tutorials/application/source_zh_cn/cv/images/vit_encoder.png)

1. ViT模型中的基础结构与标准Transformer有所不同，主要在于Normalization的位置是放在Self-Attention和Feed Forward之前，其他结构如Residual Connection，Feed Forward，Normalization都如Transformer中所设计。

2. 从Transformer结构的图片可以发现，多个子encoder的堆叠就完成了模型编码器的构建，在ViT模型中，依然沿用这个思路，通过配置超参数num_layers，就可以确定堆叠层数。

3. Residual Connection，Normalization的结构可以保证模型有很强的扩展性（保证信息经过深层处理不会出现退化的现象，这是Residual Connection的作用），Normalization和dropout的应用可以增强模型泛化能力。

从以下源码中就可以清晰看到Transformer的结构。将TransformerEncoder结构和一个多层感知器（MLP）结合，就构成了ViT模型的backbone部分。

```python
from typing import Optional

from mindspore import ops as P


class FeedForward(nn.Cell):
    """
    Feed Forward layer implementation.
    Args:
        in_features (int): The dimension of input features.
        hidden_features (int): The dimension of hidden features. Default: None.
        out_features (int): The dimension of output features. Default: None
        activation (nn.Cell): Activation function which will be stacked on top of the
        normalization layer (if not None), otherwise on top of the conv layer. Default: nn.GELU.
        keep_prob (float): The keep rate, greater than 0 and less equal than 1. Default: 1.0.
    Returns:
        Tensor, output tensor.
    Examples:
        >>> ops = FeedForward(768, 3072)
    """

    def __init__(self,
                 in_features: int,
                 hidden_features: Optional[int] = None,
                 out_features: Optional[int] = None,
                 activation: nn.Cell = nn.GELU,
                 keep_prob: float = 1.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dense1 = nn.Dense(in_features, hidden_features)
        self.activation = activation()
        self.dense2 = nn.Dense(hidden_features, out_features)
        self.dropout = nn.Dropout(keep_prob)

    def construct(self, x):
        """Feed Forward construct."""
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.dropout(x)

        return x

class ResidualCell(nn.Cell):
    """
    Cell which implements Residual function:
    $$output = x + f(x)$$
    Args:
        cell (Cell): Cell needed to add residual block.
    Returns:
        Tensor, output tensor.
    Examples:
        >>> ops = ResidualCell(nn.Dense(3,4))
    """

    def __init__(self, cell):
        super().__init__()
        self.cell = cell

    def construct(self, x):
        """ResidualCell construct."""
        return self.cell(x) + x

class DropPath(nn.Cell):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, keep_prob=None, seed=0):
        super().__init__()
        self.keep_prob = 1 - keep_prob
        seed = min(seed, 0)
        self.rand = P.UniformReal(seed=seed)
        self.shape = P.Shape()
        self.floor = P.Floor()

    def construct(self, x):
        if self.training:
            x_shape = self.shape(x)
            random_tensor = self.rand((x_shape[0], 1, 1))
            random_tensor = random_tensor + self.keep_prob
            random_tensor = self.floor(random_tensor)
            x = x / self.keep_prob
            x = x * random_tensor

        return x

class TransformerEncoder(nn.Cell):
    """
    TransformerEncoder implementation.
    Args:
        dim (int): The dimension of embedding.
        num_layers (int): The depth of transformer.
        num_heads (int): The number of attention heads.
        mlp_dim (int): The dimension of MLP hidden layer.
        keep_prob (float): The keep rate, greater than 0 and less equal than 1. Default: 1.0.
        attention_keep_prob (float): The keep rate for attention. Default: 1.0.
        drop_path_keep_prob (float): The keep rate for drop path. Default: 1.0.
        activation (nn.Cell): Activation function which will be stacked on top of the
        normalization layer (if not None), otherwise on top of the conv layer. Default: nn.GELU.
        norm (nn.Cell, optional): Norm layer that will be stacked on top of the convolution
        layer. Default: nn.LayerNorm.
    Returns:
        Tensor, output tensor.
    Examples:
        >>> ops = TransformerEncoder(768, 12, 12, 3072)
    """

    def __init__(self,
                 dim: int,
                 num_layers: int,
                 num_heads: int,
                 mlp_dim: int,
                 keep_prob: float = 1.,
                 attention_keep_prob: float = 1.0,
                 drop_path_keep_prob: float = 1.0,
                 activation: nn.Cell = nn.GELU,
                 norm: nn.Cell = nn.LayerNorm):
        super().__init__()
        drop_path_rate = 1 - drop_path_keep_prob
        dpr = [i.item() for i in np.linspace(0, drop_path_rate, num_layers)]
        attn_seeds = [np.random.randint(1024) for _ in range(num_layers)]
        mlp_seeds = [np.random.randint(1024) for _ in range(num_layers)]

        layers = []
        for i in range(num_layers):
            normalization1 = norm((dim,))
            normalization2 = norm((dim,))
            attention = Attention(dim=dim,
                                  num_heads=num_heads,
                                  keep_prob=keep_prob,
                                  attention_keep_prob=attention_keep_prob)

            feedforward = FeedForward(in_features=dim,
                                      hidden_features=mlp_dim,
                                      activation=activation,
                                      keep_prob=keep_prob)

            if drop_path_rate > 0:
                layers.append(
                    nn.SequentialCell([
                        ResidualCell(nn.SequentialCell([normalization1,
                                                        attention,
                                                        DropPath(dpr[i], attn_seeds[i])])),
                        ResidualCell(nn.SequentialCell([normalization2,
                                                        feedforward,
                                                        DropPath(dpr[i], mlp_seeds[i])]))]))
            else:
                layers.append(
                    nn.SequentialCell([
                        ResidualCell(nn.SequentialCell([normalization1,
                                                        attention])),
                        ResidualCell(nn.SequentialCell([normalization2,
                                                        feedforward]))
                    ])
                )
        self.layers = nn.SequentialCell(layers)

    def construct(self, x):
        """Transformer construct."""
        return self.layers(x)
```

### ViT模型的输入

传统的Transformer结构主要用于处理自然语言领域的词向量（Word Embedding or Word Vector），词向量与传统图像数据的主要区别在于，词向量通常是1维向量进行堆叠，而图片则是二维矩阵的堆叠，多头注意力机制在处理1维词向量的堆叠时会提取词向量之间的联系也就是上下文语义，这使得Transformer在自然语言处理领域非常好用，而2维图片矩阵如何与1维词向量进行转化就成为了Transformer进军图像处理领域的一个小门槛。

在ViT模型中：

1. 通过将输入图像在每个channel上划分为16*16个patch，这一步是通过卷积操作来完成的，当然也可以人工进行划分，但卷积操作也可以达到目的同时还可以进行一次而外的数据处理；**例如一幅输入224 x 224的图像，首先经过卷积处理得到16 x 16个patch，那么每一个patch的大小就是14 x 14。**

2. 再将每一个patch的矩阵拉伸成为一个1维向量，从而获得了近似词向量堆叠的效果。**上一步得到的14 x 14的patch就转换为长度为196的向量。**

这是图像输入网络经过的第一步处理。具体Patch Embedding的代码如下所示：


```python
class PatchEmbedding(nn.Cell):
    """
    Path embedding layer for ViT. First rearrange b c (h p) (w p) -> b (h w) (p p c).
    Args:
        image_size (int): Input image size. Default: 224.
        patch_size (int): Patch size of image. Default: 16.
        embed_dim (int): The dimension of embedding. Default: 768.
        input_channels (int): The number of input channel. Default: 3.
    Returns:
        Tensor, output tensor.
    Examples:
        >>> ops = PathEmbedding(224, 16, 768, 3)
    """
    MIN_NUM_PATCHES = 4

    def __init__(self,
                 image_size: int = 224,
                 patch_size: int = 16,
                 embed_dim: int = 768,
                 input_channels: int = 3):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.conv = nn.Conv2d(input_channels, embed_dim, kernel_size=patch_size, stride=patch_size, has_bias=True)
        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()

    def construct(self, x):
        """Path Embedding construct."""
        x = self.conv(x)
        b, c, h, w = x.shape
        x = self.reshape(x, (b, c, h * w))
        x = self.transpose(x, (0, 2, 1))

        return x

```

输入图像在划分为patch之后，会经过pos_embedding 和 class_embedding两个过程。

1. class_embedding主要借鉴了BERT模型的用于文本分类时的思想，在每一个word vector之前增加一个类别值，通常是加在向量的第一位，**上一步得到的196维的向量加上class_embedding后变为197维。**

2. 增加的class_embedding是一个可以学习的参数，经过网络的不断训练，最终以输出向量的第一个维度的输出来决定最后的输出类别；**由于输入是16 x 16个patch，所以输出进行分类时是取 16 x 16个class_embedding进行分类。**

3. pos_embedding也是一组可以学习的参数，会被加入到经过处理的patch矩阵中。

4. 由于pos_embedding也是可以学习的参数，所以它的加入类似于全链接网络和卷积的bias。**这一步就是创造一个长度维197的可训练向量加入到经过class_embedding的向量中。**

实际上，pos_embedding总共有4种方案。但是经过作者的论证，只有加上pos_embedding和不加pos_embedding有明显影响，至于pos_embedding是1维还是2维对分类结果影响不大，所以，在我们的代码中，也是采用了1维的pos_embedding，由于class_embedding是加在pos_embedding之前，所以pos_embedding的维度会比patch拉伸后的维度加1。

总的而言，ViT模型还是利用了Transformer模型在处理上下文语义时的优势，将图像转换为一种“变种词向量”然后进行处理，而这样转换的意义在于，多个patch之间本身具有空间联系，这类似于一种“空间语义”，从而获得了比较好的处理效果。

### 整体构建ViT

以下代码构建了一个完整的ViT模型。


```python
from mindspore.common.initializer import Normal


def init(init_type, shape, dtype, name, requires_grad):
    initial = initializer(init_type, shape, dtype).init_data()
    return Parameter(initial, name=name, requires_grad=requires_grad)


class ViT(nn.Cell):
    """
    Vision Transformer architecture implementation.
    Args:
        image_size (int): Input image size. Default: 224.
        input_channels (int): The number of input channel. Default: 3.
        patch_size (int): Patch size of image. Default: 16.
        embed_dim (int): The dimension of embedding. Default: 768.
        num_layers (int): The depth of transformer. Default: 12.
        num_heads (int): The number of attention heads. Default: 12.
        mlp_dim (int): The dimension of MLP hidden layer. Default: 3072.
        keep_prob (float): The keep rate, greater than 0 and less equal than 1. Default: 1.0.
        attention_keep_prob (float): The keep rate for attention layer. Default: 1.0.
        drop_path_keep_prob (float): The keep rate for drop path. Default: 1.0.
        activation (nn.Cell): Activation function which will be stacked on top of the
            normalization layer (if not None), otherwise on top of the conv layer. Default: nn.GELU.
        norm (nn.Cell, optional): Norm layer that will be stacked on top of the convolution
            layer. Default: nn.LayerNorm.
        pool (str): The method of pooling. Default: 'cls'.
    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.
    Outputs:
        Tensor of shape :math:`(N, 768)`
    Raises:
        ValueError: If `split` is not 'train', "test or 'infer'.
    Supported Platforms:
        ``GPU``
    Examples:
        >>> net = ViT()
        >>> x = ms.Tensor(np.ones([1, 3, 224, 224]), ms.float32)
        >>> output = net(x)
        >>> print(output.shape)
        (1, 768)
    About ViT:
    Vision Transformer (ViT) shows that a pure transformer applied directly to sequences of image
    patches can perform very well on image classification tasks. When pre-trained on large amounts
    of data and transferred to multiple mid-sized or small image recognition benchmarks (ImageNet,
    CIFAR-100, VTAB, etc.), Vision Transformer (ViT) attains excellent results compared to state-of-the-art
    convolutional networks while requiring substantially fewer computational resources to train.
    Citation:
    .. code-block::
        @article{2020An,
        title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
        author={Dosovitskiy, A. and Beyer, L. and Kolesnikov, A. and Weissenborn, D. and Houlsby, N.},
        year={2020},
        }
    """

    def __init__(self,
                 image_size: int = 224,
                 input_channels: int = 3,
                 patch_size: int = 16,
                 embed_dim: int = 768,
                 num_layers: int = 12,
                 num_heads: int = 12,
                 mlp_dim: int = 3072,
                 keep_prob: float = 1.0,
                 attention_keep_prob: float = 1.0,
                 drop_path_keep_prob: float = 1.0,
                 activation: nn.Cell = nn.GELU,
                 norm: Optional[nn.Cell] = nn.LayerNorm,
                 pool: str = 'cls') -> None:
        super().__init__()

        #Validator.check_string(pool, ["cls", "mean"], "pool type")
        #self.image_size = image_size

        self.patch_embedding = PatchEmbedding(image_size=image_size,
                                              patch_size=patch_size,
                                              embed_dim=embed_dim,
                                              input_channels=input_channels)
        num_patches = self.patch_embedding.num_patches

        if pool == "cls":
            self.cls_token = init(init_type=Normal(sigma=1.0),
                                  shape=(1, 1, embed_dim),
                                  dtype=ms.float32,
                                  name='cls',
                                  requires_grad=True)
            self.pos_embedding = init(init_type=Normal(sigma=1.0),
                                      shape=(1, num_patches + 1, embed_dim),
                                      dtype=ms.float32,
                                      name='pos_embedding',
                                      requires_grad=True)
            self.concat = ops.Concat(axis=1)
        else:
            self.pos_embedding = init(init_type=Normal(sigma=1.0),
                                      shape=(1, num_patches, embed_dim),
                                      dtype=ms.float32,
                                      name='pos_embedding',
                                      requires_grad=True)
            self.mean = ops.ReduceMean(keep_dims=False)

        self.pool = pool
        self.pos_dropout = nn.Dropout(keep_prob)
        self.norm = norm((embed_dim,))
        self.tile = ops.Tile()
        self.transformer = TransformerEncoder(dim=embed_dim,
                                              num_layers=num_layers,
                                              num_heads=num_heads,
                                              mlp_dim=mlp_dim,
                                              keep_prob=keep_prob,
                                              attention_keep_prob=attention_keep_prob,
                                              drop_path_keep_prob=drop_path_keep_prob,
                                              activation=activation,
                                              norm=norm)

    def construct(self, x):
        """ViT construct."""
        x = self.patch_embedding(x)

        if self.pool == "cls":
            cls_tokens = self.tile(self.cls_token, (x.shape[0], 1, 1))
            x = self.concat((cls_tokens, x))
            x += self.pos_embedding
        else:
            x += self.pos_embedding
        x = self.pos_dropout(x)
        x = self.transformer(x)
        x = self.norm(x)

        if self.pool == "cls":
            x = x[:, 0]
        else:
            x = self.mean(x, (1, 2))  # (1,) or (1,2)
        return x
```

整体流程图如下所示：

![data-process](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.9/tutorials/application/source_zh_cn/cv/images/data_process.png)

## 整体模型微调

### 预训练模型加载
我们使用`mindcv.models.vit`中定义vit_b_32_224网络，当接口中的`pretrained`参数设置为True时，可以自动下载网络权重。
由于该预训练模型是针对ImageNet数据集中的1000个类别进行分类的，这里我们设定`num_classes=2`, ViT的classifier(即最后的FC层)输出调整为两维，此时只加载backbone的预训练权重，而classifier则使用初始值。


```python
import mindspore as ms
import mindspore.nn as nn
from mindspore.nn import LossBase
from mindspore import LossMonitor, TimeMonitor

from mindcv import create_model, create_loss, create_optimizer, create_scheduler

# 定义超参
epoch_size = 5
momentum = 0.9
step_size = loader_train.get_dataset_size()

# 加载模型以及模型的预训练权重
network = create_model("vit_b_32_224", pretrained=True, num_classes=2)
```

```text
1058726912B [00:24, 43416880.85B/s]                                 
[WARNING] ME(88581:281472875162304,MainProcess):2022-12-29-11:24:34.242.783 [mindspore/train/serialization.py:712] For 'load_param_into_net', 2 parameters in the 'net' are not loaded, because they are not in the 'parameter_dict', please check whether the network structure is consistent when training and loading checkpoint.
[WARNING] ME(88581:281472875162304,MainProcess):2022-12-29-11:24:34.246.124 [mindspore/train/serialization.py:714] head.classifier.weight is not loaded.
[WARNING] ME(88581:281472875162304,MainProcess):2022-12-29-11:24:34.247.234 [mindspore/train/serialization.py:714] head.classifier.bias is not loaded.
```

### 模型训练

使用已加载处理好的带标签的狼和狗图像，对ViT进行微调网络。 
注意，对整体模型做微调时，应使用较小的learning rate。

```python
# 定义学习率策略
lr = create_scheduler(steps_per_epoch=step_size,
                      scheduler="cosine_decay",
                      lr=0.00005,
                      warmup_epochs=0,
                      num_epochs=epoch_size,
                      decay_epochs=5)

# 定义优化器
network_opt = create_optimizer(network.trainable_params(), opt="adam", lr=lr, momentum=momentum)

#define loss function
network_loss = create_loss(name="CE", reduction="mean", label_smoothing=0.1)

# initialize model
ascend_target = (ms.get_context("device_target") == "Ascend")

if ascend_target:
    model = ms.Model(network, loss_fn=network_loss, optimizer=network_opt, metrics={"acc"}, amp_level="O2")
else:
    model = ms.Model(network, loss_fn=network_loss, optimizer=network_opt, metrics={"acc"}, amp_level="O0")

# train model
model.train(epoch_size,
            loader_train,
            callbacks=[LossMonitor(15), TimeMonitor(15)],
            dataset_sink_mode=False)
```

```text
[WARNING] DEVICE(88581,ffff82bceac0,python):2022-12-29-11:25:17.084.390 [mindspore/ccsrc/plugin/device/ascend/hal/device/kernel_select_ascend.cc:330] FilterRaisedOrReducePrecisionMatchedKernelInfo] Operator:[Default/network-WithLossCell/_loss_fn-CrossEntropySmooth/GatherD-op2845] don't support int64, reduce precision from int64 to int32.
[WARNING] DEVICE(88581,ffff82bceac0,python):2022-12-29-11:25:17.204.072 [mindspore/ccsrc/plugin/device/ascend/hal/device/kernel_select_ascend.cc:330] FilterRaisedOrReducePrecisionMatchedKernelInfo] Operator:[Gradients/Default/network-WithLossCell/_backbone-BaseClassifier/backbone-ViT/gradStridedSlice/StridedSliceGrad-op2862] don't support int64, reduce precision from int64 to int32.
[WARNING] DEVICE(88581,ffff82bceac0,python):2022-12-29-11:25:44.814.641 [mindspore/ccsrc/plugin/device/ascend/hal/device/ascend_stream_assign.cc:1944] InsertEventForCallCommSubGraph] Cannot find comm group for sub comm graph label id 2


epoch: 1 step: 15, loss is 0.60576254
Train epoch time: 63755.679 ms, per step time: 4250.379 ms
epoch: 2 step: 15, loss is 0.4067704
Train epoch time: 794.584 ms, per step time: 52.972 ms
epoch: 3 step: 15, loss is 0.27751905
Train epoch time: 740.840 ms, per step time: 49.389 ms
epoch: 4 step: 15, loss is 0.2518569
Train epoch time: 746.632 ms, per step time: 49.775 ms
epoch: 5 step: 15, loss is 0.23055708
Train epoch time: 755.093 ms, per step time: 50.340 ms
```

### 模型评估

在训练完成后，我们加载验证集评估模型的精度。

```python
dataset_val = create_dataset(root=data_dir, split='val', num_parallel_workers=num_workers)

trans_val = create_transforms(dataset_name='ImageNet',is_training=False)

loader_val = create_loader(
        dataset=dataset_val,
        batch_size=5,
        is_training=False,
        num_classes=2,
        transform=trans_val,
        num_parallel_workers=num_workers,
    )

result = model.eval(loader_val)
print(result)
```

```text
{'acc': 1.0}


[WARNING] DEVICE(88581,ffff82bceac0,python):2022-12-29-11:27:04.991.915 [mindspore/ccsrc/plugin/device/ascend/hal/device/kernel_select_ascend.cc:330] FilterRaisedOrReducePrecisionMatchedKernelInfo] Operator:[Default/network-WithLossCell/_loss_fn-CrossEntropySmooth/GatherD-op5235] don't support int64, reduce precision from int64 to int32.
```

从结果可以看出，由于我们加载了预训练模型参数，模型的精度达到了100%，。如果未使用预训练模型参数，则需要更多的epoch来训练。

#### 可视化模型推理结果

定义 `visualize_mode` 函数，可视化模型预测。


```python
import numpy as np
import matplotlib.pyplot as plt

def visualize_model(model, val_dl, num_classes=2):
    # 加载验证集的数据进行验证
    images, labels= next(val_dl.create_tuple_iterator())
    # 预测图像类别
    output = model.predict(images)
    pred = np.argmax(output.asnumpy(), axis=1)
    # 显示图像及图像的预测值
    images = images.asnumpy()
    labels = labels.asnumpy()
    class_name = {0: "dogs", 1: "wolves"}
    plt.figure(figsize=(15, 7))
    for i in range(len(labels)):
        plt.subplot(3, 6, i + 1)
        # 若预测正确，显示为蓝色；若预测错误，显示为红色
        color = 'blue' if pred[i] == labels[i] else 'red'
        plt.title('predict:{}'.format(class_name[pred[i]]), color=color)
        picture_show = np.transpose(images[i], (1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        picture_show = std * picture_show + mean
        picture_show = np.clip(picture_show, 0, 1)
        plt.imshow(picture_show)
        plt.axis('off')

    plt.show()
```

```python
visualize_model(model, loader_val)
```
    
![png](output_27_0.png)
    


## 总结

本案例完成了一个ViT模型在ImageNet数据上进行训练，验证和推理的过程，其中，对关键的ViT模型结构和原理作了讲解。通过学习本案例，理解源码可以帮助用户掌握Multi-Head Attention，TransformerEncoder，pos_embedding等关键概念，如果要详细理解ViT的模型原理，建议基于源码更深层次的详细阅读。

## 引用

[1] Dosovitskiy, Alexey, et al. \"An image is worth 16x16 words: Transformers for image recognition at scale.\" arXiv preprint arXiv:2010.11929 (2020).

[2] Vaswani, Ashish, et al. \"Attention is all you need.\"Advances in Neural Information Processing Systems. (2017).
