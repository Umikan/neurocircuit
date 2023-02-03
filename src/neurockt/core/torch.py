from .base import Mapping, TensorTypes, TensorType
from abc import abstractclassmethod
from torch.nn import Module
import torch.nn as nn


class Placeholder(nn.Module):
    def __init__(self, noop, *args, **kwargs):
        super().__init__()
        self.model = noop(*args, **kwargs)
        self.__args = args
        self.__kwargs = kwargs

    def replace(self, module):
        T = self.model
        assert T.mapping() == module.mapping(
        ), f"Mapping Type Mismatch: {T.mapping()} <-> {module.mapping()}"
        self.model = module(*self.__args, **self.__kwargs)

    def forward(self, *args):
        return self.model(*args)

    def instance_of(self, cls):
        return isinstance(self.model, cls)


class TorchMapping(Module, TensorTypes):
    def __init__(self):
        Module.__init__(self)

    def replace(self, cls1, cls2):
        def replace_cls1(model):
            count = 0
            if isinstance(model, Placeholder):
                if model.instance_of(cls1):
                    model.replace(cls2)
                    count += 1
            count += sum([replace_cls1(ch) for ch in model.children()])
            return count

        count = replace_cls1(self)
        print(f"{count} {cls1.__name__} was replaced with {cls2.__name__}.")

    @classmethod
    def mapping(cls) -> Mapping:
        return Mapping(cls.input(), cls.output())

    @abstractclassmethod
    def input(cls) -> TensorType:
        raise NotImplementedError

    @abstractclassmethod
    def output(cls) -> TensorType:
        raise NotImplementedError


class Encoder(TorchMapping):
    @classmethod
    def output(cls):
        return cls.Vector


class Decoder(TorchMapping):
    @classmethod
    def input(cls):
        return cls.Vector


class Transform(TorchMapping):
    @classmethod
    def output(cls):
        return cls.input()
