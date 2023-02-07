import torch.nn as nn


class Residual(nn.Module):
    def __init__(self, *sublayers, last=None):
        super().__init__()
        self.sublayers = nn.Sequential(*sublayers)
        self.last = last

    def forward(self, x):
        x_0 = x
        for f in self.sublayers:
            x = f(x)

        x = x_0 + x
        return x if not self.last else self.last(x)


class Stack:
    def __init__(self, layer_cls, activation=None):
        self.layer_cls = layer_cls
        self.__stack = []
        self.activation = None

    def __lshift__(self, params):
        self.__stack.append(params)
        return self

    def __rshift__(self, layer_cls):
        blocks, layers = [], []
        for params in self.__stack:
            if isinstance(params, tuple):
                layer = self.layer_cls(*params)
                layers.append(layer)
                if self.activation:
                    layers.append(self.activation)
            else:
                conn = params
                blocks.append(conn(*layers))
                layers = []

        return layer_cls(*blocks, *layers)
