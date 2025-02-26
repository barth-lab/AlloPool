from jaxtyping import (
    Float,
    Int,
    Bool,
)

from torch import Tensor


class TorchTyping:
    def __init__(self, abstract_dtype):
        self.abstract_dtype = abstract_dtype

    def __getitem__(self, shapes: str):
        return self.abstract_dtype[Tensor, shapes]


Float = TorchTyping(Float)
Int = TorchTyping(Int)
Bool = TorchTyping(Bool)
