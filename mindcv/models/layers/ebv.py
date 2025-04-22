"""EBV
Mindspore implementations of Equiangular Basis Vectors layer.
Papers:
Equiangular Basis Vectors (https://arxiv.org/pdf/2303.11637)
"""

import mindspore as ms
import mindspore.numpy as msnp
from mindspore import Tensor, nn, ops


class EBV(nn.Cell):
    """
    Equiangular Basis Vectors layer
    """

    def __init__(
            self,
            num_cls: int = 1000,
            dim: int = 1000,
            thre: float = 0.002,
            slice_size: int = 130,
            lr: float = 1e-3,
            steps: int = 100000,
            tau: float = 0.07
    ) -> None:
        """
        Args:
            num_cls (int): Number of categories, which can also be interpreted as the
                        number of basis vectors N that need to be generated, num_cls >= N.  Default: 1000.
            dim (int): Dimension for basis vectors.  Default: 1000.
            thre (float): The maximum value of the absolute cosine value
                        of the angle between any two basis vectors.  Default: 0.002.
            slice_size (int): Slicing optimization is required due to insufficient memory.  Default: 130.
            lr (float): Optimize learning rate. Default: 1e-3.
            steps (int): Optimize step numbers.   Default: 100000.
            tau (float): Temperature parameter, less than
                        -num_cls/((num_cls-1) * log(exp(0.001) -1)/(N-1))). Default: 0.07
        """
        super().__init__()
        self.num_cls = num_cls
        self.dim = dim
        self.thre = thre
        self.slice_size = slice_size
        self.lr = lr
        self.steps = steps
        self.tau = tau
        self.l2norm = ops.L2Normalize()
        self.ebv = self._generate_ebv()
        self.ebv.requires_grad = False

    def _generate_ebv(self):
        basis_vec = ms.Parameter(
            ops.L2Normalize(1)(
                ops.standard_normal((self.num_cls, self.dim))
            ), name='basis_vec', requires_grad=True)
        optim = nn.SGD([basis_vec], learning_rate=self.lr)
        matmul = ops.MatMul(transpose_b=True)

        def forward_fn(a, b, e, thr):
            m = matmul(a, b).abs() - e
            loss = ops.relu(m - thr).sum()
            return loss, m

        grad_fn = ops.value_and_grad(forward_fn, 1, [basis_vec], has_aux=True)
        for _ in range(self.steps):
            basis_vec.set_data(ops.L2Normalize(1)(basis_vec.data))
            mx = self.thre
            grads = msnp.zeros_like(basis_vec)
            for i in range((self.num_cls - 1) // self.slice_size + 1):
                start = self.slice_size * i
                end = min(self.slice_size * (i + 1), self.num_cls)
                e = ops.one_hot(msnp.arange(start, end), self.num_cls, Tensor(1.0, ms.float32), Tensor(0.0, ms.float32))
                (loss, m), grads_partial = grad_fn(basis_vec[start:end], basis_vec, e, self.thre)
                mx = max(mx, m.max().asnumpy().tolist())
                grads = grads + grads_partial[0]

            if mx <= self.thre + 0.0001:
                return self.l2norm(basis_vec.data)
            optim((grads,))

        return self.l2norm(basis_vec.data)

    def construct(self, x: Tensor) -> Tensor:
        x = self.l2norm(x)
        logits = ops.matmul(x, self.ebv.T / self.tau)

        return logits
