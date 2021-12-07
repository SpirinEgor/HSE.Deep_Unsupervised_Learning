from typing import Tuple

import torch
from torch import nn


class Vae32x32(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int):
        super().__init__()
        self._latent_dim = latent_dim
        self._encoder = nn.Sequential(
            # [16; 16; 32]
            nn.Conv2d(in_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.ReLU(),
            # [8; 8; 64]
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.ReLU(),
            # [4; 4; 128]
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.ReLU(),
            # [2; 2; 256]
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.ReLU(),
            # [1; 1; 512]
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.ReLU(),
            # [1; 1; 2 * latent dim]
            nn.Conv2d(512, 2 * latent_dim, kernel_size=(1, 1), stride=(1, 1), padding=0),
        )
        self._decoder = nn.Sequential(
            # [1; 1; 512]
            nn.Conv2d(latent_dim, 512, kernel_size=(1, 1), stride=(1, 1), padding=0),
            nn.ReLU(),
            # [2; 2; 256]
            nn.ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            # [4; 4; 128]
            nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            # [8; 8; 64]
            nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            # [16; 16; 32]
            nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            # [32; 32; in channels]
            nn.ConvTranspose2d(32, in_channels, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
        )

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        # [bs; latent dim]
        mu, log_sigma = self.encode(images)

        z = self.reparameterize(mu, log_sigma)
        new_x = self.decode(z)
        return new_x, (z, mu, log_sigma)

    def encode(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, log_sigma = torch.chunk(self._encoder(images), 2, dim=1)
        return mu, log_sigma

    @staticmethod
    def reparameterize(mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(log_sigma) * torch.exp(log_sigma)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self._decoder(z)

    @torch.no_grad()
    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(n, self._latent_dim, 1, 1, device=device)
        samples = torch.clip(self.decode(z), -1, 1)  # 0, 1
        return samples.permute(0, 2, 3, 1)
