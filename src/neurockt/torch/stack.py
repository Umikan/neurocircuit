import torch


class Stack:
    def __init__(self):
        self.values = {}

    def add(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.values:
                self.values[k] = []
            self.values[k].append(v)

    def __call__(self, name):
        return torch.cat(self.values[name], dim=0).cpu()
