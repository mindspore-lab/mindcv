import numpy as np

import mindspore as ms
from mindspore import mint, nn


class PatchDropout(nn.Cell):
    """
    https://arxiv.org/abs/2212.00794
    """
    def __init__(
            self,
            prob: float = 0.5,
            num_prefix_tokens: int = 1,
            ordered: bool = False,
            return_indices: bool = False,
    ):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob
        self.num_prefix_tokens = num_prefix_tokens  # exclude CLS token (or other prefix tokens)
        self.ordered = ordered
        self.return_indices = return_indices

    def forward(self, x):
        if not self.training or self.prob == 0.:
            if self.return_indices:
                return x, None
            return x

        if self.num_prefix_tokens:
            prefix_tokens, x = x[:, :self.num_prefix_tokens], x[:, self.num_prefix_tokens:]
        else:
            prefix_tokens = None

        B = x.shape[0]
        L = x.shape[1]
        num_keep = max(1, int(L * (1. - self.prob)))
        _, indices = mint.sort(ms.Tensor(np.random.rand(B, L)).astype(ms.float32))
        keep_indices = indices[:, :num_keep]
        if self.ordered:
            # NOTE does not need to maintain patch order in typical transformer use,
            # but possibly useful for debug / visualization
            keep_indices, _ = mint.sort(keep_indices)
        keep_indices = mint.broadcast_to(mint.unsqueeze(keep_indices, dim=-1), (-1, -1, x.shape[2]))
        x = mint.gather(x, dim=1, index=keep_indices)

        if prefix_tokens is not None:
            x = mint.concat((prefix_tokens, x), dim=1)

        if self.return_indices:
            return x, keep_indices
        return x
