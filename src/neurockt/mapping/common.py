import torch.nn as nn
from ..core.torch import Transform


__all__ = [
    'FeedForward',
    'Projection',
    'ClassifierHead',
]


class FeedForward(Transform):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        return self.mlp(x)

    @classmethod
    def input(cls):
        return cls.Vector


class Projection(Transform):
    def __init__(self, proj_dim, *in_dim):
        super().__init__()
        self.proj = nn.ModuleList([nn.Linear(dim, proj_dim) for dim in in_dim])

    def forward(self, *inputs):
        return [f(input) for f, input in zip(self.proj, inputs)]

    @classmethod
    def input(cls):
        return +cls.Vector


class ClassifierHead(Transform):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(p=0.25),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.5),
            nn.Linear(512, out_features)
        )

    def forward(self, x):
        return self.net(x)

    @classmethod
    def input(cls):
        return cls.Vector
