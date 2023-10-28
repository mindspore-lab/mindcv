from collections import OrderedDict
from typing import Tuple, Union

import numpy as np

import mindspore as ms
from mindspore import Parameter, Tensor, load_param_into_net, nn, ops
from mindspore.ops.function.nn_func import multi_head_attention_forward


class Bottleneck(nn.Cell):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(
            inplanes, planes, 1, has_bias=False, pad_mode="pad", weight_init="uniform", bias_init="uniform"
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(
            planes, planes, 3, padding=1, has_bias=False, pad_mode="pad", weight_init="uniform", bias_init="uniform"
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU()

        self.avgpool = nn.AvgPool2d(kernel_size=stride, stride=stride, pad_mode="pad") if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(
            planes,
            planes * self.expansion,
            1,
            has_bias=False,
            pad_mode="pad",
            weight_init="uniform",
            bias_init="uniform",
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU()

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

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Cell):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = Parameter(
            ops.randn(spacial_dim**2 + 1, embed_dim, dtype=ms.float32) / embed_dim**0.5
        )
        self.k_proj = nn.Dense(embed_dim, embed_dim)
        self.q_proj = nn.Dense(embed_dim, embed_dim)
        self.v_proj = nn.Dense(embed_dim, embed_dim)
        self.c_proj = nn.Dense(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def construct(self, x):
        x = ops.flatten(x, start_dim=2).permute((2, 0, 1))  # NCHW -> (HW)NC
        x = ops.cat([x.mean(axis=0, keep_dims=True), x], axis=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = multi_head_attention_forward(
            query=x[:1],
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
        return ops.squeeze(x, 0)


class ModifiedResNet(nn.Cell):
    """
    A ResNet class that contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(
            3,
            width // 2,
            kernel_size=3,
            stride=2,
            padding=1,
            has_bias=False,
            pad_mode="pad",
            weight_init="uniform",
            bias_init="uniform",
        )
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            width // 2,
            width // 2,
            kernel_size=3,
            padding=1,
            has_bias=False,
            pad_mode="pad",
            weight_init="uniform",
            bias_init="uniform",
        )
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(
            width // 2,
            width,
            kernel_size=3,
            padding=1,
            has_bias=False,
            pad_mode="pad",
            weight_init="uniform",
            bias_init="uniform",
        )
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU()
        self.avgpool = nn.AvgPool2d(kernel_size=2, pad_mode="pad", stride=2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.SequentialCell(*layers)

    def construct(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.to(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class QuickGELU(nn.Cell):
    def construct(self, x: Tensor):
        return x * ops.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Cell):
    def __init__(self, d_model: int, n_head: int, attn_mask: Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm([d_model], epsilon=1e-5)
        self.mlp = nn.SequentialCell(
            OrderedDict(
                [
                    ("c_fc", nn.Dense(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Dense(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = nn.LayerNorm([d_model], epsilon=1e-5)
        self.attn_mask = Parameter(attn_mask) if attn_mask is not None else None

    def attention(self, x: Tensor):
        if self.attn_mask is not None:
            return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask.to(x.dtype))[0]
        else:
            return self.attn(x, x, x, need_weights=False)[0]

    def construct(self, x: Tensor):
        x_type = x.dtype
        x = x + self.attention(self.ln_1(x.to(ms.float32)).to(x_type))
        x = x + self.mlp(self.ln_2(x.to(ms.float32)).to(x_type))
        return x


class Transformer(nn.Cell):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.SequentialCell(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def construct(self, x: Tensor):
        return self.resblocks(x)


class VisionTransformer(nn.Cell):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            has_bias=False,
            pad_mode="pad",
            weight_init="uniform",
            bias_init="uniform",
        )

        scale = width**-0.5
        self.class_embedding = Parameter(scale * ops.randn(width))
        self.positional_embedding = Parameter(scale * ops.randn(((input_resolution // patch_size) ** 2 + 1, width)))
        self.ln_pre = nn.LayerNorm([width], epsilon=1e-5)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = nn.LayerNorm([width], epsilon=1e-5)
        self.proj = Parameter(scale * ops.randn((width, output_dim)))

    def construct(self, x: Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape((x.shape[0], x.shape[1], -1))  # shape = [*, width, grid ** 2]
        x = x.permute((0, 2, 1))  # shape = [*, grid ** 2, width]
        x = ops.cat(
            [self.class_embedding.to(x.dtype) + ops.zeros((x.shape[0], 1, x.shape[-1]), dtype=x.dtype), x], axis=1
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x_type = x.dtype
        x = self.ln_pre(x.to(ms.float32)).to(x_type)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x_type = x.dtype
        x = self.ln_post(x.to(ms.float32)[:, 0, :]).to(x_type)

        if self.proj is not None:
            x = x @ self.proj

        return x


class CLIP(nn.Cell):
    def __init__(
        self,
        embed_dim: int,
        # vision
        image_resolution: int,
        vision_layers: Union[Tuple[int, int, int, int], int],
        vision_width: int,
        vision_patch_size: int,
        # text
        context_length: int,
        vocab_size: int,
        transformer_width: int,
        transformer_heads: int,
        transformer_layers: int,
    ):
        super().__init__()

        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width,
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = Parameter(ms.numpy.empty((self.context_length, transformer_width)))
        self.ln_final = nn.LayerNorm([transformer_width], epsilon=1e-5)

        self.text_projection = Parameter(ms.numpy.empty((transformer_width, embed_dim)))
        self.logit_scale = Parameter(ops.ones(()) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        self.token_embedding.embedding_table.set_data(
            ops.normal(self.token_embedding.embedding_table.shape, stddev=0.02, mean=0)
        )
        self.positional_embedding.set_data(ops.normal(self.positional_embedding.shape, stddev=0.01, mean=0))

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_channels**-0.5
                self.visual.attnpool.q_proj.weight.set_data(
                    ops.normal(self.visual.attnpool.q_proj.weight.shape, stddev=std, mean=0)
                )
                self.visual.attnpool.k_proj.weight.set_data(
                    ops.normal(self.visual.attnpool.k_proj.weight.shape, stddev=std, mean=0)
                )
                self.visual.attnpool.v_proj.weight.set_data(
                    ops.normal(self.visual.attnpool.v_proj.weight.shape, stddev=std, mean=0)
                )
                self.visual.attnpool.c_proj.weight.set_data(
                    ops.normal(self.visual.attnpool.c_proj.weight.shape, stddev=std, mean=0)
                )

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for param in resnet_block.get_parameters():
                    if param.name.endswith("bn3.weight"):
                        param.set_data(ops.zeros(param.shape))

        proj_std = (self.transformer.width**-0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width**-0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            block.attn.in_proj_weight.set_data(ops.normal(block.attn.in_proj_weight.shape, stddev=attn_std, mean=0))
            block.attn.out_proj.weight.set_data(ops.normal(block.attn.out_proj.weight.shape, stddev=proj_std, mean=0))
            block.mlp.c_fc.weight.set_data(ops.normal(block.mlp.c_fc.weight.shape, stddev=fc_std, mean=0))
            block.mlp.c_proj.weight.set_data(ops.normal(block.mlp.c_proj.weight.shape, stddev=proj_std, mean=0))

        if self.text_projection is not None:
            self.text_projection.set_data(
                ops.normal(self.text_projection.shape, stddev=self.transformer.width**-0.5, mean=0)
            )

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        mask = ops.fill(ms.float32, (self.context_length, self.context_length), float("-inf"))
        mask = mask.triu(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.to(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).to(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x.to(ms.float32)).to(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[ops.arange(x.shape[0]), text.argmax(axis=-1)] @ self.text_projection

        return x

    def construct(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Cell):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(layer):
        if isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Dense)):
            layer.weight.to(ms.float16)
            if layer.bias is not None:
                layer.bias.to(ms.float16)

        if isinstance(layer, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                param = getattr(layer, attr)
                if param is not None:
                    param.to(ms.float16)

        for name in ["text_projection", "proj"]:
            if hasattr(layer, name):
                attr = getattr(layer, name)
                if attr is not None:
                    attr.to(ms.float16)

    model.apply(_convert_weights_to_fp16)


def build_model(ckpt_dict: dict):
    vit = "visual.proj" in ckpt_dict

    if vit:
        vision_width = ckpt_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in ckpt_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")]
        )
        vision_patch_size = ckpt_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((ckpt_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [
            len(set(k.split(".")[2] for k in ckpt_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]
        ]
        vision_layers = tuple(counts)
        vision_width = ckpt_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((ckpt_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width**2 + 1 == ckpt_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = ckpt_dict["text_projection"].shape[1]
    context_length = ckpt_dict["positional_embedding"].shape[0]
    vocab_size = ckpt_dict["token_embedding.embedding_table"].shape[0]
    transformer_width = ckpt_dict["ln_final.gamma"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in ckpt_dict if k.startswith("transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution,
        vision_layers,
        vision_width,
        vision_patch_size,
        context_length,
        vocab_size,
        transformer_width,
        transformer_heads,
        transformer_layers,
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in ckpt_dict:
            del ckpt_dict[key]

    convert_weights(model)
    load_param_into_net(model, ckpt_dict)
    return model.set_train(False)
