import torch.nn as nn
from ..core.torch import TorchMapping


__all__ = [
    'GlobalAveragePooling'
]


class GlobalAveragePooling(TorchMapping):
    def __init__(self):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.avgpool(x)
        x = self.flatten(x)
        return x

    @classmethod
    def input(cls):
        return cls.Image

    @classmethod
    def output(cls):
        return cls.Vector
