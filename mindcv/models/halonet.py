"""
MindSpore implementation of `HaloNet`.
Refer to Scaling Local Self-Attention for Parameter Effificient Visual Backbones.
"""
import mindspore as ms
import mindspore.common.initializer as init
import mindspore.numpy as msnp
from mindspore import Parameter, Tensor, nn, ops
from mindspore.common.initializer import HeUniform, TruncatedNormal

from .helpers import load_pretrained, make_divisible
from .layers.identity import Identity
from .registry import register_model

__all__ = [
    "halonet_50t",
    "HaloNet"
]


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 256, 256),
        "first_conv": "stem.conv1.conv",
        "classifier": "classifier",
        **kwargs,
    }


default_cfgs = {
    "halonet_50t": _cfg(
        url="https://download.mindspore.cn/toolkits/mindcv/halonet/halonet_50t-533da6be.ckpt")
}


class ConvBnAct(nn.Cell):
    """ Build layer contain: conv - bn - act
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 act=None,
                 bias_init=False
                 ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              pad_mode="pad",
                              padding=padding,
                              weight_init=HeUniform(),
                              has_bias=bias_init
                              )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = ActLayer(act)

    def construct(self, inputs):
        out = self.conv(inputs)
        out = self.bn(out)
        out = self.act(out)
        return out


class ActLayer(nn.Cell):
    """ Build Activation Layer according to act type
    """
    def __init__(self, act):
        super().__init__()
        if act == 'silu':
            self.act = nn.SiLU()
        elif act == 'relu':
            self.act = nn.ReLU()
        else:
            self.act = Identity()

    def construct(self, inputs):
        out = self.act(inputs)
        return out


class BatchNormAct2d(nn.Cell):
    """ Build layer contain: bn-act
    """
    def __init__(self, chs, act=None):
        super().__init__()
        self.bn = nn.BatchNorm2d(chs)
        self.act = ActLayer(act)

    def construct(self, inputs):
        out = self.bn(inputs)
        out = self.act(out)
        return out


class SelectAdaptivePool2d(nn.Cell):
    """ Selectable global pooling layer with dynamic input kernel size
    """
    def __init__(self, pool_type='avg', flatten=False):
        super().__init__()
        # convert other false values to empty string for consistent TS typing
        self.pool_type = pool_type or ''
        self.flatten = nn.Flatten() if flatten else Identity()
        if pool_type == '':
            self.pool = Identity()
        elif pool_type == 'avg':
            self.pool = ops.ReduceMean(keep_dims=True)
        else:
            assert False, 'Invalid pool type: %s' % pool_type

    def construct(self, inputs):
        out = self.pool(inputs, (2, 3))
        out = self.flatten(out)
        return out


class Stem(nn.Cell):
    def __init__(self, act):
        super().__init__()
        self.conv1 = ConvBnAct(3, 24, kernel_size=3, stride=2, padding=1, act=act)
        self.conv2 = ConvBnAct(24, 32, kernel_size=3, stride=1, padding=1, act=act)
        self.conv3 = ConvBnAct(32, 64, kernel_size=3, stride=1, padding=1, act=act)
        self.pad = ops.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='valid')

    def construct(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pad(x)
        x = self.pool(x)
        return x


def rel_logits_1d(q, rel_k, permute_mask):
    """ Compute relative logits along one dimension
    :param q: [batch,H,W,dim]
    :param rel_k: [2*window-1,dim]
    :param permute_mask: permute output axis according to this
    """
    B, H, W, _ = q.shape
    rel_size = rel_k.shape[0]
    win_size = (rel_size+1)//2
    rel_k = ops.transpose(rel_k, (1, 0))
    x = msnp.tensordot(q, rel_k, axes=1)
    x = ops.reshape(x, (-1, W, rel_size))
    # pad to shift from relative to absolute indexing
    x_pad = ops.pad(x, paddings=((0, 0), (0, 0), (0, 1)))
    x_pad = ops.flatten(x_pad)
    x_pad = ops.expand_dims(x_pad, 1)
    x_pad = ops.pad(x_pad, paddings=((0, 0), (0, 0), (0, rel_size - W)))
    x_pad = ops.squeeze(x_pad, axis=())
    # reshape adn slice out the padded elements
    x_pad = ops.reshape(x_pad, (-1, W+1, rel_size))
    x = x_pad[:, :W, win_size-1:]
    # reshape and tile
    x = ops.reshape(x, (B, H, 1, W, win_size))
    x = ops.broadcast_to(x, (B, H, win_size, W, win_size))
    x = ops.transpose(x, permute_mask)
    return x


class RelPosEmb(nn.Cell):
    """ Relative Position Embedding
    """
    def __init__(
            self,
            block_size,
            win_size,
            dim_head,
            ):
        """
        :param block_size (int): block size
        :param win_size (int): neighbourhood window size
        :param dim_head (int): attention head dim
        :param scale (float): scale factor (for init)
        """
        super().__init__()
        self.block_size = block_size
        self.dim_head = dim_head
        tensor1 = Tensor(shape=((2 * win_size - 1), dim_head), dtype=ms.float32, init=TruncatedNormal(sigma=.02))
        self.rel_height = Parameter(tensor1)
        tensor2 = Tensor(shape=((2 * win_size - 1), dim_head), dtype=ms.float32, init=TruncatedNormal(sigma=.02))
        self.rel_width = Parameter(tensor2)

    def construct(self, q):
        B, BB, HW, _ = q.shape
        # relative logits in width dimension
        q = ops.reshape(q, (-1, self.block_size, self.block_size, self.dim_head))
        rel_logits_w = rel_logits_1d(q, self.rel_width, permute_mask=(0, 1, 3, 2, 4))
        # relative logits in height dimension
        q = ops.transpose(q, (0, 2, 1, 3))
        rel_logits_h = rel_logits_1d(q, self.rel_height, permute_mask=(0, 3, 1, 4, 2))
        rel_logits = rel_logits_h+rel_logits_w
        rel_logits = ops.reshape(rel_logits, (B, BB, HW, -1))
        return rel_logits


class HaloAttention(nn.Cell):
    """
    The internal dimensions of the attention module are controlled by
    the interaction of several arguments.
    the output dimension : dim_out
    the value(v) dimension :  dim_out//num_heads
    the query(q) and key(k) dimensions are determined by :
         * num_heads*dim_head
         * num_heads*(dim_out*attn_ratio//num_heads)
    the ratio of q and k relative to the output : attn_ratio

    Args:
        dim (int): input dimension to the module
        dim_out (int): output dimension of the module, same as dim if not set
        feat_size (Tuple[int, int]): size of input feature_map (not used, for arg compat with bottle/lambda)
        stride: output stride of the module, query downscaled if > 1 (default: 1).
        num_heads: parallel attention heads (default: 8).
        dim_head: dimension of query and key heads, calculated from dim_out * attn_ratio // num_heads if not set
        block_size (int): size of blocks. (default: 8)
        halo_size (int): size of halo overlap. (default: 3)
        qk_ratio (float): ratio of q and k dimensions to output dimension when dim_head not set. (default: 1.0)
        qkv_bias (bool) : add bias to q, k, and v projections
        avg_down (bool): use average pool downsample instead of strided query blocks
        scale_pos_embed (bool): scale the position embedding as well as Q @ K
    """
    def __init__(self,
                 dim,
                 dim_out=None,
                 feat_size=None,
                 stride=1,
                 num_heads=8,
                 dim_head=None,
                 block_size=8,
                 halo_size=3,
                 qk_ratio=1.0,  # ratio of q and k dimensions to output dimension when dim_head not set.
                 qkv_bias=False,
                 avg_down=False,  # use average pool downsample instead of strided query blocks
                 scale_pos_embed=False):  # scale the position embedding as well as Q @ K
        super().__init__()
        dim_out = dim_out or dim
        assert dim_out % num_heads == 0
        self.stride = stride
        self.num_heads = num_heads  # 8
        self.dim_head_qk = make_divisible(dim_out * qk_ratio, divisor=8) // num_heads
        self.dim_head_v = dim_out // self.num_heads  # dimension of head
        self.dim_out_qk = num_heads * self.dim_head_qk
        self.dim_out_v = num_heads * self.dim_head_v  # dimension of dim_out_v
        self.scale = self.dim_head_qk ** -0.5
        self.scale_pos_embed = scale_pos_embed
        self.block_size = self.block_size_ds = block_size
        self.halo_size = halo_size
        self.win_size = block_size + halo_size * 2  # neighbourhood window size
        self.block_stride = stride
        use_avg_pool = False
        if stride > 1:
            use_avg_pool = avg_down or block_size % stride != 0
            self.block_stride = stride
            self.block_size_ds = self.block_size // self.block_stride
        self.q = nn.Conv2d(dim,
                           self.dim_out_qk,
                           1,
                           stride=self.block_stride,
                           has_bias=qkv_bias,
                           weight_init=HeUniform())
        self.kv = nn.Conv2d(dim, self.dim_out_qk + self.dim_out_v, 1, has_bias=qkv_bias)
        self.pos_embed = RelPosEmb(
            block_size=self.block_size_ds, win_size=self.win_size, dim_head=self.dim_head_qk)
        self.pool = nn.AvgPool2d(2, 2) if use_avg_pool else Identity()
        self.softmax_fn = ops.Softmax(-1)
        self.pad_kv = ops.Pad(
            paddings=((0, 0), (0, 0), (self.halo_size, self.halo_size), (self.halo_size, self.halo_size))
            )
        self.kv_unfold = nn.Unfold(
            ksizes=[1, self.win_size, self.win_size, 1],
            strides=[1, self.block_size, self.block_size, 1],
            rates=[1, 1, 1, 1],
            padding='valid'
            )

    def construct(self, x):
        B, C, H, W = x.shape
        assert H % self.block_size == 0 and W % self.block_size == 0, 'fmap dimensions must be divisible'
        num_h_blocks = H//self.block_size
        num_w_blocks = W//self.block_size
        num_blocks = num_h_blocks * num_w_blocks
        q = self.q(x)
        # unfold
        q = ops.reshape(q, (-1, self.dim_head_qk, num_h_blocks, self.block_size_ds, num_w_blocks, self.block_size_ds))
        q = ops.transpose(q, (0, 1, 3, 5, 2, 4))
        q = ops.reshape(q, (B*self.num_heads, self.dim_head_qk, -1, num_blocks))
        q = ops.transpose(q, (0, 3, 2, 1))  # B*num_heads,num_blocks,block_size**2, dim_head
        kv = self.kv(x)  # [bs,dim_out,H,W]
        kv = self.pad_kv(kv)
        kv = self.kv_unfold(kv)  # B, C_kh_kw, _, _
        kv = ops.reshape(kv, (B * self.num_heads, self.dim_head_qk + self.dim_head_v, -1, num_blocks))
        kv = ops.transpose(kv, (0, 3, 2, 1))  # [B * self.num_heads, num_blocks, -1, self.dim_head_qk + self.dim_head_v]
        k = kv[..., :self.dim_head_qk]
        v = kv[..., self.dim_head_qk:(self.dim_head_qk + self.dim_head_v)]
        k = ops.transpose(k, (0, 1, 3, 2))  # [B * self.num_heads, num_blocks, self.dim_head_qk, -1]
        if self.scale_pos_embed:
            attn = (ops.matmul(q, k) + self.pos_embed(q)) * self.scale
        else:
            pos_embed_q = self.pos_embed(q)
            part_1 = (ops.matmul(q, k)) * self.scale
            attn = part_1 + pos_embed_q
        # attn: B * num_heads, num_blocks, block_size ** 2, win_size ** 2
        attn = self.softmax_fn(attn)
        # attn = attn @ v
        attn = ops.matmul(attn, v)  # attn: B * num_heads, num_blocks, block_size ** 2, dim_head_v
        out = ops.transpose(attn, (0, 3, 2, 1))  # B * num_heads, dim_head_v, block_size ** 2, num_blocks
        # fold
        out = ops.reshape(out, (-1, self.block_size_ds, self.block_size_ds, num_h_blocks, num_w_blocks))
        # -1, num_h_blocks, self.block_size_ds, num_w_blocks, self.block_size_ds
        out = ops.transpose(out, (0, 3, 1, 4, 2))
        out = ops.reshape(out, (B, self.dim_out_v, H // self.block_stride, W // self.block_stride))
        # B, dim_out, H // block_stride, W // block_stride
        out = self.pool(out)
        return out


class BottleneckBlock(nn.Cell):
    """ ResNet-like Bottleneck Block - 1x1 - kxk - 1x1
    """
    def __init__(self,
                 in_chs,
                 out_chs,
                 stride,
                 act,
                 downsample=None,
                 shortcut=None,
                 ):
        super().__init__()
        self.stride = stride
        mid_chs = out_chs//4
        self.conv1_1x1 = ConvBnAct(
                                   in_chs,
                                   mid_chs,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0,
                                   act=act)
        self.conv2_kxk = ConvBnAct(
                                   mid_chs,
                                   mid_chs,
                                   kernel_size=3,
                                   stride=self.stride,
                                   padding=1,
                                   act=act)
        self.conv2b_kxk = Identity()
        self.conv3_1x1 = ConvBnAct(
                                   mid_chs,
                                   out_chs,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0)
        self.attn = Identity()
        self.attn_last = Identity()
        self.shortcut = shortcut
        if self.shortcut:
            if downsample:
                self.creat_shortcut = ConvBnAct(
                                                in_chs,
                                                out_chs,
                                                kernel_size=1,
                                                stride=self.stride,
                                                padding=0)
            else:
                self.creat_shortcut = ConvBnAct(
                                                in_chs,
                                                out_chs,
                                                kernel_size=1,
                                                stride=1,
                                                padding=0)
        self.Identity = Identity()
        self.act = ActLayer(act)

    def construct(self, x):
        h = x
        x = self.conv1_1x1(x)
        x = self.conv2_kxk(x)
        x = self.conv2b_kxk(x)
        x = self.attn(x)
        x = self.conv3_1x1(x)
        out = self.attn_last(x)
        if self.shortcut:
            h = self.creat_shortcut(h)
        else:
            h = self.Identity(h)
        out = out + h
        out = self.act(out)
        return out


class SelfAttnBlock(nn.Cell):
    """ ResNet-like Bottleneck Block - 1x1 -kxk - self attn -1x1
    """
    def __init__(self,
                 chs,
                 num_heads,
                 block_size,
                 halo_size,
                 act,
                 stride=None,
                 shortcut=None,
                 hidden_chs=None,
                 ):
        super().__init__()
        mid_chs = chs//4
        if hidden_chs is None:
            out_chs = chs
        else:
            out_chs = hidden_chs

        if stride is None:
            self.stride = 1
        else:
            self.stride = stride
        self.conv1_1x1 = ConvBnAct(out_chs, mid_chs, kernel_size=1, stride=1, padding=0, act=act)
        self.conv2_kxk = Identity()
        self.conv3_1x1 = ConvBnAct(mid_chs, chs, kernel_size=1, stride=1, padding=0)
        self.self_attn = HaloAttention(mid_chs,
                                       dim_out=mid_chs,
                                       block_size=block_size,
                                       halo_size=halo_size,
                                       num_heads=num_heads,
                                       stride=self.stride)
        self.post_attn = BatchNormAct2d(mid_chs, act=act)
        self.shortcut = shortcut
        if self.shortcut:
            self.creat_shortcut = ConvBnAct(
                                            out_chs,
                                            chs,
                                            kernel_size=1,
                                            stride=self.stride,
                                            padding=0)
        self.Identity = Identity()
        self.act = ActLayer(act=act)

    def construct(self, x):
        h = x
        out = self.conv1_1x1(x)
        out = self.self_attn(out)
        out = self.post_attn(out)
        out = self.conv3_1x1(out)
        if self.shortcut:
            h = self.creat_shortcut(h)
        else:
            h = self.Identity(h)
        out = out + h
        out = self.act(out)
        return out


class HaloStage(nn.Cell):
    """ Stage layers for HaloNet. Stage layers contains a number of Blocks.
    """
    def __init__(self,
                 block_types,
                 block_size,
                 halo_size,
                 depth,
                 channel,
                 out_channel,
                 stride,
                 num_head,
                 act,
                 hidden_chs=None,
                 downsample=None,
                 ):
        super().__init__()
        self.depth = depth
        blocks = []
        for idx in range(depth):
            if idx == 0:
                shortcut = True
                in_channel = channel
                if downsample is None:
                    self.down = False
                else:
                    self.down = downsample
                block_stride = stride
                self.hidden = hidden_chs
            else:
                stride = 1
                shortcut = False
                in_channel = out_channel
                self.down = False
                block_stride = 1
                self.hidden = None

            block_type = block_types[idx]
            if block_type == 'bottle':
                blocks.append(
                    BottleneckBlock(
                        in_chs=in_channel,
                        out_chs=out_channel,
                        stride=block_stride,
                        shortcut=shortcut,
                        downsample=self.down,
                        act=act,
                    )
                )
            if block_type == 'attn':
                if num_head > 0:
                    blocks.append(
                        SelfAttnBlock(
                            chs=out_channel,
                            stride=stride,
                            num_heads=num_head,
                            block_size=block_size,
                            halo_size=halo_size,
                            hidden_chs=self.hidden,
                            shortcut=shortcut,
                            act=act,
                        )
                    )
        self.blocks = nn.CellList(blocks)

    def construct(self, x):
        for stage in self.blocks:
            x = stage(x)
        return x


class HaloNet(nn.Cell):
    """ Define main structure of HaloNet: stem - blocks - head
    """
    def __init__(self,
                 depth_list,
                 block_size,
                 halo_size,
                 stage1_block,
                 stage2_block,
                 stage3_block,
                 stage4_block,
                 chs_list,
                 num_heads,
                 num_classes,
                 stride_list,
                 hidden_chs,
                 act,
                 ):
        super().__init__()
        self.stem = Stem(act)
        self.stage1 = HaloStage(
                                block_types=stage1_block,
                                block_size=block_size,
                                halo_size=halo_size,
                                depth=depth_list[0],
                                channel=chs_list[0],
                                out_channel=chs_list[1],
                                stride=stride_list[0],
                                num_head=num_heads[0],
                                hidden_chs=hidden_chs,
                                act=act,
                                )
        self.stage2 = HaloStage(
                                block_types=stage2_block,
                                block_size=block_size,
                                halo_size=halo_size,
                                depth=depth_list[1],
                                channel=chs_list[1],
                                out_channel=chs_list[2],
                                stride=stride_list[1],
                                num_head=num_heads[1],
                                hidden_chs=hidden_chs,
                                act=act,
                                downsample=True)
        self.stage3 = HaloStage(
                                block_types=stage3_block,
                                block_size=block_size,
                                halo_size=halo_size,
                                depth=depth_list[2],
                                channel=chs_list[2],
                                out_channel=chs_list[3],
                                stride=stride_list[2],
                                num_head=num_heads[2],
                                hidden_chs=hidden_chs,
                                act=act,
                                downsample=True)
        self.stage4 = HaloStage(
                                block_types=stage4_block,
                                block_size=block_size,
                                halo_size=halo_size,
                                depth=depth_list[3],
                                channel=chs_list[3],
                                out_channel=chs_list[4],
                                stride=stride_list[3],
                                num_head=num_heads[3],
                                hidden_chs=hidden_chs,
                                act=act,
                                downsample=True)
        self.classifier = nn.SequentialCell([
            SelectAdaptivePool2d(flatten=True),
            nn.Dense(chs_list[4], num_classes, TruncatedNormal(.02), bias_init='zeros'),
            Identity()]
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize weights for cells."""
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(init.initializer(init.HeUniform(), cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(init.initializer("zeros", cell.bias.shape, cell.bias.dtype))
            elif isinstance(cell, nn.BatchNorm2d):
                cell.gamma.set_data(init.initializer("ones", cell.gamma.shape, cell.gamma.dtype))
                cell.beta.set_data(init.initializer("zeros", cell.beta.shape, cell.beta.dtype))
            elif isinstance(cell, nn.Dense):
                cell.weight.set_data(init.initializer(init.HeUniform(), cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(init.initializer("zeros", cell.bias.shape, cell.bias.dtype))

    def construct(self, x):
        x = self.stem(x)
        out_stage1 = self.stage1(x)
        out_stage2 = self.stage2(out_stage1)
        out_stage3 = self.stage3(out_stage2)
        out_stage4 = self.stage4(out_stage3)
        out = self.classifier(out_stage4)
        return out


@register_model
def halonet_50t(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    """Get HaloNet model.
    Refer to the base class `models.HaloNet` for more details."""
    default_cfg = default_cfgs["halonet_50t"]
    model = HaloNet(
        depth_list=[3, 4, 6, 3],
        stage1_block=['bottle', 'bottle', 'bottle'],
        stage2_block=['bottle', 'bottle', 'bottle', 'attn'],
        stage3_block=['bottle', 'attn', 'bottle', 'attn', 'bottle', 'attn'],
        stage4_block=['bottle', 'attn', 'bottle'],
        chs_list=[64, 256, 512, 1024, 2048],
        num_heads=[0, 4, 8, 8],
        num_classes=num_classes,
        stride_list=[1, 2, 2, 2],
        block_size=8,
        halo_size=3,
        hidden_chs=None,
        act='silu',
        **kwargs
    )
    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model
