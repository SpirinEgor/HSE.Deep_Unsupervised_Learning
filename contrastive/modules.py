import torch
from torch import nn
from torch.nn import functional as F


class Block(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        bias: bool = False,
        upsample: bool = False,
        lr_cf: float = 0.2,
    ):
        super().__init__()
        self.upsample = upsample
        self.conv = nn.Conv2d(in_ch, out_ch, (3, 3), stride=stride, padding=1, bias=bias)
        self.act = nn.LeakyReLU(lr_cf)

    def forward(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False, recompute_scale_factor=False)
        return self.act(self.conv(x))


class ImageEncoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()

        self.blocks = nn.Sequential(
            Block(1, 16, stride=2),  # [1; 28; 28] -> [16; 14; 14]
            Block(16, 32, stride=2),  # [16; 14; 14] -> [32; 7; 7]
            Block(32, 64, stride=2),  # [32; 7; 7] -> [64; 4; 4]
            Block(64, 128, stride=2),  # [64; 4; 4] -> [128; 2; 2]
            Block(128, 128, stride=2).conv,  # [128; 2; 2] -> [128; 1; 1]
        )
        self.linear = nn.Linear(128, latent_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        convolved = self.blocks(images)
        return self.linear(convolved.view(images.shape[0], -1))


class Predictor(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.LeakyReLU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)