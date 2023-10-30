from collections import OrderedDict

import mindspore as ms
from mindspore import Parameter, Tensor, nn, ops
from mindspore.ops.function.nn_func import multi_head_attention_forward


class Bottleneck(nn.Cell):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=1, has_bias=False, pad_mode="pad", weight_init="uniform", bias_init="uniform"
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            padding=1,
            has_bias=False,
            pad_mode="pad",
            weight_init="uniform",
            bias_init="uniform",
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.act2 = nn.ReLU()

        self.avgpool = nn.AvgPool2d(kernel_size=stride, stride=stride, pad_mode="pad") if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(
            planes,
            planes * self.expansion,
            kernel_size=1,
            has_bias=False,
            pad_mode="pad",
            weight_init="uniform",
            bias_init="uniform",
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.act3 = nn.ReLU()

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.SequentialCell(
                OrderedDict(
                    [
                        ("9999", nn.AvgPool2d(kernel_size=stride, stride=stride, pad_mode="pad")),
                        (
                            "0",
                            nn.Conv2d(
                                inplanes,
                                planes * self.expansion,
                                1,
                                stride=1,
                                has_bias=False,
                                pad_mode="pad",
                                weight_init="uniform",
                                bias_init="uniform",
                            ),
                        ),
                        ("1", nn.BatchNorm2d(planes * self.expansion)),
                    ]
                )
            )

    def construct(self, x: Tensor) -> Tensor:
        identity = x

        out = self.act1(self.bn1(self.conv1(x).to(x.dtype)))
        out = self.act2(self.bn2(self.conv2(out).to(x.dtype)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out).to(x.dtype))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act3(out)
        return out


class AttentionPool2d(nn.Cell):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = Parameter(ops.randn((spacial_dim**2 + 1, embed_dim)) / embed_dim**0.5)
        self.k_proj = nn.Dense(embed_dim, embed_dim)
        self.q_proj = nn.Dense(embed_dim, embed_dim)
        self.v_proj = nn.Dense(embed_dim, embed_dim)
        self.c_proj = nn.Dense(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def construct(self, x):
        xtype = x.dtype
        x = x.reshape((x.shape[0], x.shape[1], x.shape[2] * x.shape[3])).permute((2, 0, 1))  # NCHW -> (HW)NC
        x = ops.cat([x.mean(axis=0, keep_dims=True), x], axis=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x = x.to(ms.float32)
        x, _ = multi_head_attention_forward(
            query=x,
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=ops.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0.0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
        )

        return x[0].to(xtype)


class ModifiedResNet(nn.Cell):
    """
    A ResNet class that contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, image_size=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.image_size = image_size

        # the 3-layer stem
        self.conv1 = nn.Conv2d(
            3, width // 2, kernel_size=3, stride=2, padding=1, has_bias=False, pad_mode="pad", weight_init="HeUniform"
        )
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            width // 2, width // 2, kernel_size=3, padding=1, has_bias=False, pad_mode="pad", weight_init="HeUniform"
        )
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(
            width // 2, width, kernel_size=3, padding=1, has_bias=False, pad_mode="pad", weight_init="HeUniform"
        )
        self.bn3 = nn.BatchNorm2d(width)
        self.act3 = nn.ReLU()
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2, pad_mode="pad")

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(image_size // 32, embed_dim, heads, output_dim)

        self.init_parameters()

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.SequentialCell(*layers)

    def init_parameters(self):
        if self.attnpool is not None:
            std = self.attnpool.c_proj.in_channels**-0.5
            self.attnpool.q_proj.weight.set_data(ops.normal(self.attnpool.q_proj.weight.shape, stddev=std, mean=0))
            self.attnpool.k_proj.weight.set_data(ops.normal(self.attnpool.k_proj.weight.shape, stddev=std, mean=0))
            self.attnpool.v_proj.weight.set_data(ops.normal(self.attnpool.v_proj.weight.shape, stddev=std, mean=0))
            self.attnpool.c_proj.weight.set_data(ops.normal(self.attnpool.c_proj.weight.shape, stddev=std, mean=0))

        for resnet_block in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for param in resnet_block.get_parameters():
                if param.name.endswith("bn3.weight"):
                    param.set_data(ops.zeros(param.shape))

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        assert unlocked_groups == 0, "partial locking not currently supported for this model"
        for param in self.get_parameters():
            param.requires_grad = False
        # freeze_batch_norm_2d realize the same operation as the above
        # "requires_grad = False" to nn.BatchNorm2d and nn.SyncBatchNorm

    def set_grad_checkpointing(self, enable=True):
        # FIXME support for non-transformer
        pass

    def stem(self, x):
        x = self.act1(self.bn1(self.conv1(x).to(x.dtype)))
        x = self.act2(self.bn2(self.conv2(x).to(x.dtype)))
        x = self.act3(self.bn3(self.conv3(x).to(x.dtype)))
        x = self.avgpool(x)
        return x

    def construct(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x
