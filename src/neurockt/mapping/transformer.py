import torch.nn as nn
from torch import einsum
import math
from .common import FeedForward, Projection
from ..core.torch import Transform, TorchMapping


__all__ = [
    'ScaleDotProductAttention',
    'TransformerBlock'
]


class ScaleDotProductAttention(TorchMapping):
    def __init__(self, dim):
        super().__init__()
        self.softmax = nn.Softmax(dim=2)
        self.scale = math.sqrt(dim)

    def forward(self, Q, K, V):
        QK = einsum("bqd,bkd->bqk", Q, K)
        weights = self.softmax(QK / self.scale)
        return einsum("bqk,bkd->bqd", weights, V)

    @classmethod
    def input(cls):
        return cls.Token.repeat(3)

    @classmethod
    def output(cls):
        return cls.Token


class TransformerBlock(Transform):
    def __init__(self, dim, q_dim, k_dim, v_dim):
        super().__init__()
        self.proj = Projection(dim, q_dim, k_dim, v_dim)
        self.attn = ScaleDotProductAttention(dim)
        self.ffn = FeedForward(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        Q, K, V = self.proj(x, x, x)
        x_ = self.attn(Q, K, V)
        x = self.norm1(x + x_)
        x_ = self.ffn(x)
        return self.norm2(x + x_)

    @classmethod
    def input(cls):
        return cls.Token
