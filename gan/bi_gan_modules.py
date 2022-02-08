import torch
from torch import nn, Tensor


class BiGanGenerator(nn.Module):
    def __init__(self, img_sz: int, latent_dim: int, hidden_dim: int = 1024):
        super().__init__()

        self._img_sz = img_sz
        self._latent = latent_dim

        self._model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, affine=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, img_sz * img_sz),
            nn.Tanh(),
        )

    def forward(self, z: Tensor) -> Tensor:
        batch_size = z.shape[0]
        out = self._model(z)
        return out.reshape(batch_size, 1, self._img_sz, self._img_sz)


class BiGanDiscriminator(nn.Module):
    def __init__(self, z_dim: int, x_dim: int, hidden_dim: int, leaky_relu_cf: float = 0.2):
        super().__init__()

        self._model = nn.Sequential(
            nn.Linear(z_dim + x_dim, hidden_dim),
            nn.LeakyReLU(leaky_relu_cf),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, affine=False),
            nn.LeakyReLU(leaky_relu_cf),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, z: Tensor, x: Tensor) -> Tensor:
        batch_size = x.shape[0]
        flatten_x = x.reshape(batch_size, -1)
        cat = torch.cat([z, flatten_x], dim=1)
        return self._model(cat)


class BiGanEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, leaky_relu_cf: float = 0.2):
        super().__init__()

        self._model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(leaky_relu_cf),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, affine=False),
            nn.LeakyReLU(leaky_relu_cf),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.shape[0]
        flatten_x = x.reshape(batch_size, -1)
        return self._model(flatten_x)
