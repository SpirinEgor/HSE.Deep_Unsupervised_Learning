import torch
from torch import nn


class Classifier(nn.Module):
    def __init__(self, latent_dim=1, hidden_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 2, 1),
            nn.ReLU(),
        )

        self.linear = nn.Sequential(
            nn.Linear(4 * 4 * 128 + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, z):
        out = self.conv(x)
        out = torch.flatten(out, start_dim=1)

        return self.linear(torch.cat((out, z), dim=1))


class Encoder(nn.Module):
    def __init__(self, latent_dim=1, noise_dim=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 2, 1),
            nn.ReLU(),
        )

        self.linear = nn.Linear(4 * 4 * 128 + noise_dim, latent_dim)

        self.noise_dim = noise_dim

    def forward(self, x, noise):
        out = self.conv(x)
        out = torch.flatten(out, start_dim=1)

        return self.linear(torch.cat((out, noise), dim=1))


class Decoder(nn.Module):
    def __init__(self, latent_dim=1):
        super().__init__()
        self.linear = nn.Sequential(nn.Linear(latent_dim, 4 * 4 * 128), nn.ReLU())

        self.convt = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 3, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, 1, 1),
        )

    def forward(self, x):
        return self.convt(self.linear(x).reshape(x.shape[0], 128, 4, 4))
