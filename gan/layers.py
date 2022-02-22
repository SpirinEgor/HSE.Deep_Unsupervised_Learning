from typing import Tuple

import torch
from torch import nn, Tensor


class UpSampleConv2d(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, kernel: Tuple[int, int] = (3, 3), stride: int = 1, padding: int = 1):
        super().__init__()
        self._conv = nn.Conv2d(in_dim, out_dim, kernel_size=kernel, stride=stride, padding=padding)

    def forward(self, input_batch: Tensor) -> Tensor:
        """Up sample forward pass

        :param input_batch: tensor of shape [B; C_in; H; W]
        :return: tensor of shape [B; C_out; H * 2; W * 2]
        """
        # [B; C_in * 4; H; W]
        input_batch = torch.cat([input_batch, input_batch, input_batch, input_batch], dim=1)
        # [B; C_in; H * 2; W * 2]
        d2s = torch.pixel_shuffle(input_batch, 2)
        # [B; C_out; H * 2; W * 2]
        out = self._conv(d2s)
        return out


class DownSampleConv2d(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, kernel: Tuple[int, int] = (3, 3), stride: int = 1, padding: int = 1):
        super().__init__()
        self._conv = nn.Conv2d(in_dim, out_dim, kernel_size=kernel, stride=stride, padding=padding)

    def forward(self, input_batch: Tensor) -> Tensor:
        """Down sample forward pass

        :param input_batch: tensor of shape [B; C_in; H * 2; W * 2]
        :return: tensor of shape [B; C_out; H; W]
        """
        # [B; C_in * 4; H; W]
        s2d = torch.pixel_unshuffle(input_batch, 2)
        # List of [B; C_in; H; W]
        chunked = torch.chunk(s2d, chunks=4, dim=1)
        # [B; C_in; H; W]
        mean = sum(chunked) / 4.0
        # [B; C_out; H; W]
        out = self._conv(mean)
        return out


class ResNetBlockUp(nn.Module):
    def __init__(self, in_dim: int, n_filters: int = 256):
        super().__init__()
        self._layers = nn.Sequential(
            *[
                nn.BatchNorm2d(in_dim),
                nn.ReLU(),
                nn.Conv2d(in_dim, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=1),
                nn.BatchNorm2d(n_filters),
                nn.ReLU(),
                UpSampleConv2d(n_filters, n_filters, kernel=(3, 3), padding=1),
            ]
        )
        self._straight_up = UpSampleConv2d(in_dim, n_filters, kernel=(1, 1), padding=0)

    def forward(self, input_batch: Tensor) -> Tensor:
        return self._layers(input_batch) + self._straight_up(input_batch)


class ResNetBlockDown(nn.Module):
    def __init__(self, in_dim: int, n_filters: int = 256):
        super().__init__()
        self._layers = nn.Sequential(
            *[
                nn.ReLU(),
                nn.Conv2d(in_dim, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=1),
                nn.ReLU(),
                DownSampleConv2d(n_filters, n_filters, kernel=(3, 3), padding=1),
            ]
        )
        self._straight_down = DownSampleConv2d(in_dim, n_filters, kernel=(1, 1), padding=0)

    def forward(self, input_batch: Tensor) -> Tensor:
        return self._layers(input_batch) + self._straight_down(input_batch)


class ResNetBlock(nn.Module):
    def __init__(self, in_dim: int, n_filters: int = 256):
        super().__init__()
        self._layers = nn.Sequential(
            *[
                nn.ReLU(),
                nn.Conv2d(in_dim, n_filters, kernel_size=(3, 3), padding=1),
                nn.ReLU(),
                nn.Conv2d(n_filters, n_filters, kernel_size=(3, 3), padding=1),
            ]
        )

    def forward(self, input_batch: Tensor) -> Tensor:
        return self._layers(input_batch) + input_batch
