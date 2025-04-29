""" MLP module w/ dropout and configurable activation layer
"""
from typing import Optional

from mindspore import Tensor, mint, nn


class Mlp(nn.Cell):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Optional[nn.Cell] = mint.nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = mint.nn.Linear(in_features, hidden_features, bias=True)
        self.act = act_layer()
        self.fc2 = mint.nn.Linear(hidden_features, out_features, bias=True)
        self.drop = mint.nn.Dropout(p=drop)

    def construct(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
