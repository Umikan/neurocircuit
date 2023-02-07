import torch
import torch.nn as nn


class EinsumAttention(nn.Module):
    def __init__(self, einsum_op, left, right):
        super().__init__()
        self.einsum_op = einsum_op
        self.left = left
        self.right = right
        self._get_name = lambda: f"Einsum<{self.__class__.__name__}>"

    def forward(self, x, y=None):
        if y is None:
            y = x
        return torch.einsum(self.einsum_op, self.left(x), self.right(y))


class SELayer(EinsumAttention):
    def __init__(self, n_channels, reduction=16, n_dim=2):
        if n_dim == 2:
            pool = nn.AdaptiveAvgPool2d(1)
            op = "bc,bcwh->bcwh"
        elif n_dim == 1:
            pool = nn.AdaptiveAvgPool1d(1)
            op = "bc,bcwh->bcl"
        else:
            assert False, "Squeeze-Excitation Layer only accepts 1D or 2D inputs."

        se = nn.Sequential(
            pool,
            nn.Flatten(),
            nn.Linear(n_channels, n_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(n_channels // reduction, n_channels, bias=False),
            nn.Sigmoid()
        )
        super().__init__(op, se, nn.Identity())


class DotProduct(EinsumAttention):
    def __init__(self, dim, q_dim, k_dim):
        super().__init__("bqd,bkd->bqk", nn.Linear(q_dim, dim), nn.Linear(k_dim, dim))
