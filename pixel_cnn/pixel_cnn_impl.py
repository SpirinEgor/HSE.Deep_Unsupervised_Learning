from typing import Tuple

import torch
from torch import nn, Tensor, softmax

from pixel_cnn.masked_conv import ConvTypeB, ConvTypeA


class ResidualBlock(nn.Module):
    def __init__(self, n_channels: int):
        super().__init__()
        self.__model = nn.Sequential(
            ConvTypeB(in_channels=n_channels, out_channels=n_channels // 2, kernel_size=1),
            nn.ReLU(),
            ConvTypeB(in_channels=n_channels // 2, out_channels=n_channels // 2, kernel_size=7, padding=3),
            nn.ReLU(),
            ConvTypeB(in_channels=n_channels // 2, out_channels=n_channels, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.__model(x) + x


class PixelCNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        height: int,
        width: int,
        n_layers: int,
        n_filters: int,
    ):
        super().__init__()
        self._c, self._h, self._w = in_channels, height, width
        self._out = out_channels

        layers = [ConvTypeA(in_channels, n_filters, kernel_size=7, padding=3)]
        for _ in range(n_layers):
            layers.append(ResidualBlock(n_filters))
        layers.extend(
            [
                nn.ReLU(),
                ConvTypeB(in_channels=n_filters, out_channels=n_filters, kernel_size=1),
                nn.ReLU(),
                ConvTypeB(in_channels=n_filters, out_channels=in_channels * self._out, kernel_size=1),
            ]
        )

        self.__model = nn.Sequential(*layers)

    def forward(self, batch: Tensor) -> Tensor:
        """Forward pass of PixelCNN

        :param batch: tensor of shape [N; C; H; W]
        :return: tensor of shape [N; OUT; C; H; W ]
        """
        # [N; C * OUT; H; W]
        output = self.__model(batch)
        return output.reshape(batch.shape[0], self._out, self._c, self._h, self._w)

    @property
    def input_shape(self) -> Tuple[int, int, int]:
        return self._c, self._h, self._w


class PixelCNNVae(PixelCNN):
    def __init__(
        self, in_size: int, in_channels: int, hidden_channels: int = 120, num_bins: int = 4, num_blocks: int = 8
    ):
        super().__init__(in_channels, num_bins // in_channels, in_size, in_size, num_blocks, hidden_channels)
        self._emb = nn.Embedding(num_bins, in_channels)
        self._num_bins = num_bins
        self._in_size = in_size

    def forward(self, batch: Tensor) -> Tensor:
        # [B, C, H, W]
        emb = self._emb(batch).squeeze(1).permute(0, 3, 1, 2)
        b, _, h, w = emb.shape
        return super().forward(emb).reshape(b, self._num_bins, h, w)

    @torch.no_grad()
    def sample(self, num_samples: int, device: torch.device) -> Tensor:
        result = torch.zeros((num_samples, self._in_size, self._in_size), dtype=torch.long).to(device)
        for i in range(self._in_size):
            for j in range(self._in_size):
                out = self(result)
                probs = softmax(out, dim=1)[..., i, j]

                result[:, i, j] = torch.multinomial(probs, num_samples=1).flatten()
        return result
