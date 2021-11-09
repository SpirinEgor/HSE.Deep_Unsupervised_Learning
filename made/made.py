from typing import List

import torch
from numpy import arange, repeat
from numpy.random import randint
from torch.nn import ReLU, Sequential, Module

from masked_linear import MaskedLinear


class MADE(Module):
    """Implements Masked AutoEncoder for Density Estimation, by Germain et al. 2015
    Paper link: https://arxiv.org/abs/1502.03509
    """

    def __init__(self, n_features: int, d_size: int, hidden_sizes: List[int] = None, order: List[int] = None):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = []
        self.__layer_sizes = [n_features] + hidden_sizes + [n_features * d_size]
        self.__d_size = d_size

        self.__layers = []
        for h_in, h_out in zip(self.__layer_sizes, self.__layer_sizes[1:]):
            self.__layers += [MaskedLinear(h_in, h_out), ReLU()]
        self.__layers.pop()
        self.__made = Sequential(*self.__layers)

        self.__init_masks(d_size, order)

    def __init_masks(self, d_size: int, order: List[int] = None):
        if order is None:
            order = arange(self.__layer_sizes[0])
        n_layers = len(self.__layers)

        prev_order = order
        for _l in range(n_layers):
            cur_order = randint(prev_order.min(), self.__layer_sizes[0] - 1, size=self.__layer_sizes[_l])
            mask = prev_order[:, None] <= cur_order[None, :]
            self.__layers[_l * 2].set_mask(mask)
            prev_order = cur_order
        last_layer_mask = prev_order[:, None] < order
        last_layer_mask = repeat(last_layer_mask[-1], d_size, axis=1)
        self.__layers[-1].set_mask(last_layer_mask)

    def forward(self, input_batch: torch.Tensor) -> torch.Tensor:
        """MADE forward pass.
        :param input_batch: [batch size; n features] tensor with input features
        :return [batch size; n features; d size] tensor with autoencoded features distributions
        """
        batch_size, _ = input_batch.shape
        return self.__made(input_batch).view(batch_size, -1, self.__d_size)
