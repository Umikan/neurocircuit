import torch
import torch.nn as nn
from ..core.torch import TorchMapping
from torch.nn.utils.rnn import pad_sequence


__all__ = [
    'MultiCategoryEmbedder'
]


class MultiCategoryEmbedder(TorchMapping):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.pad = vocab_size
        self.embeddings = nn.Embedding(
            vocab_size + 1, dim, padding_idx=self.pad)

    def forward(self, x):
        sequences = []
        for sample in x:
            tokens = (sample == 1).nonzero().squeeze(1)
            sequences.append(tokens)

        x = pad_sequence(sequences, padding_value=self.pad).transpose(0, 1)
        mask = torch.where(x != self.pad, True, False)
        return self.embeddings(x), mask

    @classmethod
    def input(cls):
        return cls.Binary

    @classmethod
    def output(cls):
        return cls.Token + cls.Binary
