import torch


class Batch:
    def __init__(self):
        self.value = []

    def __iadd__(self, other):
        self.value.append(other)
        return self

    def cat(self):
        return torch.cat(self.value, dim=0).cpu()
