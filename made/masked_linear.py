import torch
from numpy import ndarray, uint8
from torch import nn, tensor
from torch.nn.functional import linear


class MaskedLinear(nn.Linear):
    """Linear layer with configurable mask to drop weights."""

    def __init__(self, in_features: int, out_features: int, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer("mask", torch.ones(out_features, in_features))

    def set_mask(self, mask: ndarray):
        self.mask.data.copy_(torch.from_numpy(mask.astype(uint8).T))

    def forward(self, data: tensor):
        return linear(data, self.mask * self.weight, self.bias)
