from abc import abstractmethod

from torch import Tensor, zeros_like
from torch.nn import Conv2d, Parameter


class MaskedConv(Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask = Parameter(self.get_mask())

    @abstractmethod
    def get_mask(self) -> Tensor:
        raise NotImplementedError()

    def forward(self, *args, **kwargs):
        self.weight.data *= self.mask
        return super().forward(*args, **kwargs)


class ConvTypeA(MaskedConv):
    def get_mask(self) -> Tensor:
        mask = zeros_like(self.weight)
        h, w = self.kernel_size
        mask[:, :, : h // 2] = 1
        mask[:, :, h // 2, : w // 2] = 1
        return mask


class ConvTypeB(MaskedConv):
    def get_mask(self) -> Tensor:
        mask = zeros_like(self.weight)
        h, w = self.kernel_size
        mask[:, :, : h // 2] = 1
        mask[:, :, h // 2, : w // 2 + 1] = 1
        return mask
