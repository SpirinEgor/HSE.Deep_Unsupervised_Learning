from typing import Tuple

import torch
from torch import nn, Tensor


class ResidualBlock2d(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self._layers = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(input_dim),
            nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(input_dim),
            nn.Conv2d(input_dim, input_dim, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self._layers(x)


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z: Tensor) -> Tuple[Tensor, ...]:
        emb_w = self.embedding.weight

        with torch.no_grad():
            # (x - y)^2 = x^2 + y^2 - 2xy
            distance = torch.sum(z ** 2, dim=1, keepdim=True) + torch.sum(emb_w ** 2, dim=1) - 2 * z @ emb_w.t()
            indices = torch.argmin(distance, dim=-1)

        quantized = self.embedding(indices)

        return quantized, (quantized - z).detach() + z, indices


class VqVae(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, num_embeddings: int, embedding_dim: int):
        super().__init__()

        self._num_emb = num_embeddings
        self._emb_dim = embedding_dim

        self._quantizer = VectorQuantizer(num_embeddings, embedding_dim)
        self._encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            ResidualBlock2d(hidden_dim),
            ResidualBlock2d(hidden_dim),
        )
        self._decoder = nn.Sequential(
            ResidualBlock2d(hidden_dim),
            ResidualBlock2d(hidden_dim),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_dim),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_dim),
            nn.ConvTranspose2d(hidden_dim, in_channels, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        # [B, H, W, C]
        z = self._encoder(x).permute(0, 2, 3, 1)
        # [B * H * W, C]
        z_flat = z.reshape(-1, self._emb_dim)

        e, e_straight, indices = self._quantizer(z_flat)

        # [B, C, H, W]
        x_recon = self._decoder(e_straight.reshape(z.shape).permute(0, 3, 1, 2))
        return x_recon, z_flat, e

    @torch.no_grad()
    def encode_to_code(self, x: Tensor) -> Tensor:
        emb = self._encoder(x).permute(0, 2, 3, 1)
        _, _, indices = self._quantizer(emb.reshape(-1, self._emb_dim))
        return indices.reshape(emb.shape[0], 1, emb.shape[1], emb.shape[2])

    @torch.no_grad()
    def decode_from_code(self, code: Tensor) -> Tensor:
        emb = self._quantizer.embedding(code.reshape(code.shape[0], -1))
        emb = emb.reshape(code.shape[0], code.shape[2], code.shape[3], self._emb_dim).permute(0, 3, 1, 2)
        reconstructed = self._decoder(emb)
        return reconstructed
