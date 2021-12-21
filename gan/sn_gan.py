import torch.distributions
from torch import nn, Tensor

from gan.layers import ResNetBlockUp, ResNetBlockDown


class SNGanGenerator128to32x32(nn.Module):
    def __init__(self, noise: torch.distributions.Distribution, n_filters: int = 256):
        super().__init__()
        self._noise = noise
        self._fc = nn.Linear(128, 4 * 4 * 256)
        self._generator = nn.Sequential(
            *[
                ResNetBlockUp(in_dim=256, n_filters=n_filters),  # [B; 256; 4; 4] -> [B; # filters; 8; 8]
                ResNetBlockUp(in_dim=n_filters, n_filters=n_filters),  # -> [B; # filters; 16; 16]
                ResNetBlockUp(in_dim=n_filters, n_filters=n_filters),  # -> [B; # filters; 32; 32]
                nn.BatchNorm2d(n_filters),
                nn.ReLU(),
                nn.Conv2d(n_filters, 3, kernel_size=(3, 3), padding=1),  # -> [B; 3; 32; 32]
                nn.Tanh(),
            ]
        )

    def forward(self, input_z: Tensor) -> Tensor:
        z = self._fc(input_z).reshape(-1, 256, 4, 4)
        return self._generator(z)

    def sample(self, n_samples: int) -> Tensor:
        z = self._noise.sample([n_samples, 128])
        return self(z)


class SNGanDiscriminator32x32to128(nn.Module):
    def __init__(self, n_filters: int = 256):
        super().__init__()
        self._fc = nn.Linear(128, 1)
        self._discriminator = nn.Sequential(
            *[
                ResNetBlockDown(3, n_filters),  # [B; 3; 32; 32] -> [B; # filters; 16; 16]
                ResNetBlockDown(n_filters, n_filters),  # -> [B; # filters; 8; 8]
                ResNetBlockDown(n_filters, n_filters),  # -> [B; # filters; 4; 4]
                nn.Conv2d(n_filters, 128, kernel_size=(3, 3), padding=1),  # -> [B; 128; 4; 4]
                nn.ReLU(),
            ]
        )

    def forward(self, input_batch: Tensor) -> Tensor:
        # [B; 128; 4; 4]
        z = self._discriminator(input_batch)
        # [B; 128]
        z = torch.sum(z, dim=(2, 3))
        # [B; 1]
        return self.fc(z)
