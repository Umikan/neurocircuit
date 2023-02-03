class TensorType:
    def __init__(self, value: str):
        self.dims = value

    def data_dim(self):
        return sum([symbol.isdecimal()
                    for symbol in set(self.dims.split(" "))])

    def __eq__(self, other):
        return self.dims == other.dims

    def __add__(self, other):
        return TensorType(f"{self.dims} . {other.dims}")

    def repeat(self, n):
        out = self
        for _ in range(n - 1):
            out = self + out
        return out

    def __pos__(self):
        return TensorType(f"[{self.dims}]")

    def __str__(self):
        return self.dims


class TensorTypes:
    Image = TensorType("d 0 1")
    Token = TensorType("0 d")
    Vector = TensorType("v")
    Binary = TensorType("b")


class Mapping:
    def __init__(self, in_tensor: TensorType, out_tensor: TensorType):
        self.in_tensor = in_tensor
        self.out_tensor = out_tensor

    def __eq__(self, other):
        return self.in_tensor == other.in_tensor and self.out_tensor == other.out_tensor

    def io(self):
        return f"{self.in_tensor} -> {self.out_tensor}"
