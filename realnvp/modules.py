from typing import Tuple

import torch
from torch import nn


class NormalizedConv2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.conv = nn.utils.weight_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.conv(batch)


class ActNorm(nn.Module):
    def __init__(self, n_channels: int):
        super().__init__()
        self.n_channels = n_channels

        self.log_s = nn.Parameter(torch.zeros(1, n_channels, 1, 1), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(1, n_channels, 1, 1), requires_grad=True)
        self._initialized = False

    def forward(self, batch: torch.Tensor, reverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        if reverse:
            return (batch - self.b) * torch.exp(-self.log_s), self.log_s

        if not self._initialized:
            self.b.data = -torch.mean(batch, dim=(0, 2, 3), keepdim=True)
            s = torch.std(batch.permute(1, 0, 2, 3).reshape(self.n_channels, -1), dim=1).reshape(
                1, self.n_channels, 1, 1
            )
            self.log_s.data = -torch.log(s)

        return batch * self.log_s.exp() + self.b, self.log_s


class ResidualBlock(nn.Module):
    def __init__(self, n_filters: int):
        super().__init__()
        self.block = nn.Sequential(
            NormalizedConv2D(n_filters, n_filters, 1, 1, 0),
            nn.ReLU(),
            NormalizedConv2D(n_filters, n_filters, 3, 1, 1),
            nn.ReLU(),
            NormalizedConv2D(n_filters, n_filters, 1, 1, 0),
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return batch + self.block(batch)


class AffineCouplingWithCheckerboard(nn.Module):
    def __init__(self, top_left: int = 1, img_sz: int = 32, n_blocks: int = 4, hidden_ch: int = 128):
        super().__init__()
        self.mask = torch.arange(img_sz) + torch.arange(img_sz).reshape(-1, 1)
        self.mask = torch.remainder(top_left + self.mask, 2)
        self.mask = self.mask.reshape(1, 1, img_sz, img_sz).float()
        self.mask = nn.Parameter(self.mask, requires_grad=False)

        self.g = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(1), requires_grad=True)

        layers = [NormalizedConv2D(3, hidden_ch, 3, 1, 1)]
        for _ in range(n_blocks):
            layers.append(ResidualBlock(hidden_ch))
        layers += [nn.ReLU(), NormalizedConv2D(hidden_ch, 6, 3, 1, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, batch: torch.Tensor, reverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        bs, n_channels, *_ = batch.shape
        mask = self.mask.repeat(bs, 1, 1, 1)

        masked_batch = batch * mask

        log_s, b = self.net(masked_batch).split(n_channels, dim=1)
        log_s = self.g * torch.tanh(log_s) + self.b

        b = b * (1 - mask)
        log_s = log_s * (1 - mask)

        x = ((batch - b) * torch.exp(-log_s)) if reverse else (batch * log_s.exp() + b)
        return x, log_s


class AffineCouplingWithChannels(nn.Module):
    def __init__(self, is_top_left: bool = True, n_blocks: int = 4, hidden_ch: int = 128):
        super().__init__()
        self.is_top_left = is_top_left

        self.g = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(1), requires_grad=True)

        layers = [NormalizedConv2D(6, hidden_ch, 3, 1, 1)]
        for _ in range(n_blocks):
            layers.append(ResidualBlock(hidden_ch))
        layers += [nn.ReLU(), NormalizedConv2D(hidden_ch, 12, 3, 1, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, batch: torch.Tensor, reverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        bs, n_channels, *_ = batch.shape

        on, off = batch.split(n_channels // 2, dim=1)
        if not self.is_top_left:
            on, off = off, on

        log_s, b = self.net(off).split(n_channels // 2, dim=1)
        log_s = self.g * torch.tanh(log_s) + self.b

        on = ((on - b) * torch.exp(-log_s)) if reverse else on * log_s.exp() + b

        if self.is_top_left:
            return torch.cat([on, off], dim=1), torch.cat([log_s, torch.zeros_like(log_s)], dim=1)
        else:
            return torch.cat([off, on], dim=1), torch.cat([torch.zeros_like(log_s), log_s], dim=1)


class RealNVP(nn.Module):
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.checker_transforms_a = nn.ModuleList(
            [
                AffineCouplingWithCheckerboard(1, hidden_ch=hidden_dim),
                ActNorm(3),
                AffineCouplingWithCheckerboard(0, hidden_ch=hidden_dim),
                ActNorm(3),
                AffineCouplingWithCheckerboard(1, hidden_ch=hidden_dim),
                ActNorm(3),
                AffineCouplingWithCheckerboard(0, hidden_ch=hidden_dim),
            ]
        )
        self.checker_transforms_b = nn.ModuleList(
            [
                AffineCouplingWithCheckerboard(1, hidden_ch=hidden_dim),
                ActNorm(3),
                AffineCouplingWithCheckerboard(0, hidden_ch=hidden_dim),
                ActNorm(3),
                AffineCouplingWithCheckerboard(1, hidden_ch=hidden_dim),
            ]
        )
        self.channel_transforms = nn.ModuleList(
            [
                AffineCouplingWithChannels(True, hidden_ch=hidden_dim),
                ActNorm(12),
                AffineCouplingWithChannels(False, hidden_ch=hidden_dim),
                ActNorm(12),
                AffineCouplingWithChannels(True, hidden_ch=hidden_dim),
            ]
        )

    def forward(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        log_det = torch.zeros_like(batch)

        for i, trans in enumerate([self.checker_transforms_a, self.channel_transforms, self.checker_transforms_b]):
            for t in trans:
                batch, d = t(batch)
                log_det += d

            if i == 0:
                batch, log_det = squeeze(batch), squeeze(log_det)
            elif i == 1:
                batch, log_det = unsqueeze(batch), unsqueeze(log_det)

        return batch, log_det

    def generate(self, z: torch.Tensor) -> torch.Tensor:
        for i, trans in enumerate([self.checker_transforms_b, self.channel_transforms, self.checker_transforms_a]):
            for t in reversed(trans):
                z, _ = t(z, True)

            if i == 0:
                z = squeeze(z)
            elif i == 1:
                z = unsqueeze(z)
        return z


def squeeze(x: torch.Tensor) -> torch.Tensor:
    b, c, h, w = x.shape
    x = x.reshape(b, c, h // 2, 2, w // 2, 2)
    x = x.permute(0, 1, 3, 5, 2, 4)
    x = x.reshape(b, c * 4, h // 2, w // 2)
    return x


def unsqueeze(x: torch.Tensor) -> torch.Tensor:
    b, c, h, w = x.shape
    x = x.reshape(b, c // 4, 2, 2, h, w)
    x = x.permute(0, 1, 4, 2, 5, 3)
    x = x.reshape(b, c // 4, h * 2, w * 2)
    return x
