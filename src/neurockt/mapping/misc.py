from einops import rearrange
from ..core.torch import TorchMapping


__all__ = [
    'ImageToToken'
]


class ImageToToken(TorchMapping):
    def forward(self, x):
        self.w = x.size(2)
        return rearrange(x, "b c w h -> b (w h) c")

    def reverse(self, x):
        return rearrange(x, "b (w h) c -> b c w h", w=self.w)

    @classmethod
    def input(cls):
        return cls.Image

    @classmethod
    def output(cls):
        return cls.Token
