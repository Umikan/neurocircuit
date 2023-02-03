import torch.nn as nn
from neurockt.core import TorchMapping, Placeholder


class WithoutActivation(TorchMapping):
    def __init__(self, dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, dim),
            nn.Linear(dim, dim),
            nn.Linear(dim, n_classes)
        )

    def forward(self, x):
        return self.net(x)

    @classmethod
    def input(cls):
        return cls.Vector

    @classmethod
    def output(cls):
        return cls.Vector


class WithActivation(TorchMapping):
    def __init__(self, dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, dim),
            Placeholder(nn.ReLU),
            nn.Linear(dim, dim),
            Placeholder(nn.ReLU),
            nn.Linear(dim, n_classes)
        )

    def forward(self, x):
        return self.net(x)


class Net(TorchMapping):
    def __init__(self, dim, n_classes):
        super().__init__()
        self.net = Placeholder(WithActivation, dim, n_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        return self.net(x)

    @classmethod
    def input(cls):
        return cls.Vector

    @classmethod
    def output(cls):
        return cls.Vector
