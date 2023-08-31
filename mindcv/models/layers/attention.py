"""attention layers
TODO: add Flash Attention
"""

from typing import Optional

from mindspore import Tensor, nn, ops

from .compatibility import Dropout


class Attention(nn.Cell):
    """
    Attention layer implementation, Rearrange Input -> B x N x hidden size.

    Args:
        dim (int): The dimension of input features.
        num_heads (int): The number of attention heads. Default: 8.
        qkv_bias (bool): Specifies whether the linear layer uses a bias vector. Default: True.
        qk_scale: (float): The user-defined factor to scale the product of q and k. Default: None.
        attn_drop (float): The drop rate of attention, greater than 0 and less equal than 1. Default: 0.0.
        proj_drop (float): The drop rate of output, greater than 0 and less equal than 1. Default: 0.0.
        attn_head_dim (int): The user-defined dimension of attention head features. Default: None.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> ops = Attention(768, 12)
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        attn_head_dim: Optional[int] = None,
    ):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * num_heads

        if qk_scale:
            self.scale = Tensor(qk_scale)
        else:
            self.scale = Tensor(head_dim ** -0.5)

        self.qkv = nn.Dense(dim, all_head_dim * 3, has_bias=qkv_bias)

        self.attn_drop = Dropout(attn_drop)
        self.proj = nn.Dense(all_head_dim, dim)
        self.proj_drop = Dropout(proj_drop)

        self.mul = ops.Mul()
        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()
        self.unstack = ops.Unstack(axis=0)
        self.attn_matmul_v = ops.BatchMatMul()
        self.q_matmul_k = ops.BatchMatMul(transpose_b=True)
        self.softmax = nn.Softmax(axis=-1)

    def construct(self, x, rel_pos_bias=None):
        b, n, c = x.shape
        qkv = self.qkv(x)
        qkv = self.reshape(qkv, (b, n, 3, self.num_heads, c // self.num_heads))
        qkv = self.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = self.unstack(qkv)

        attn = self.q_matmul_k(q, k)
        attn = self.mul(attn, self.scale)

        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        out = self.attn_matmul_v(attn, v)
        out = self.transpose(out, (0, 2, 1, 3))
        out = self.reshape(out, (b, n, c))
        out = self.proj(out)
        out = self.proj_drop(out)

        return out
