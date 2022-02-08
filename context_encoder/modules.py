import torch
import torch.nn.functional as F
from torch import nn


class DenoisingBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        bias: bool = False,
        upsample: bool = False,
        lr_cf: float = 0.2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.upsample = upsample
        self.conv = nn.Conv2d(in_ch, out_ch, (3, 3), stride=stride, padding=1, bias=bias)
        self.norm = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(lr_cf)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False, recompute_scale_factor=False)
        x = x + torch.randn_like(x) * 0.05
        x = self.dropout(x)
        return self.act(self.norm(self.conv(x)))


class MaskedImageEncoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()

        self.blocks = nn.Sequential(
            DenoisingBlock(1, 16, stride=2),  # [1; 28; 28] -> [16; 14; 14]
            DenoisingBlock(16, 32, stride=2),  # [16; 14; 14] -> [32; 7; 7]
            DenoisingBlock(32, 64, stride=2),  # [32; 7; 7] -> [64; 4; 4]
            DenoisingBlock(64, 128, stride=2),  # [64; 4; 4] -> [128; 2; 2]
            DenoisingBlock(128, 128, stride=2).conv,  # [128; 2; 2] -> [128; 1; 1]
        )
        self.linear = nn.Linear(128, latent_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        convolved = self.blocks(images)
        return self.linear(convolved.view(images.shape[0], -1))


class PatchDecoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.linear = nn.Linear(latent_dim, 128 * 2 * 2)
        self.blocks = nn.Sequential(
            DenoisingBlock(128, 64, upsample=True),  # [128; 2; 2] -> [64; 4; 4]
            DenoisingBlock(64, 32, upsample=True),  # [64; 4; 4] -> [32; 8; 8]
            DenoisingBlock(32, 16, upsample=True),  # [32; 8; 8] -> [16; 16; 16]
            nn.Conv2d(16, 1, kernel_size=(3, 3)),  # [16; 16; 16] -> [1; 14; 14]
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        sized = self.linear(images)
        sized = sized.view(-1, 128, 2, 2)
        decoded = self.blocks(sized)
        return torch.sigmoid(decoded)


class PatchDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.blocks = nn.Sequential(
            DenoisingBlock(1, 16, stride=2),  # [1; 14; 14] -> [16; 7; 7]
            DenoisingBlock(16, 32, stride=2),  # [16; 7; 7] -> [32; 7; 7]
            DenoisingBlock(32, 64, stride=2),  # [32; 4; 4] -> [64; 2; 2]
            DenoisingBlock(64, 128, stride=2),  # [64; 2; 2] -> [128; 1; 1]
        )
        self.linear = nn.Linear(128, 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        convolved = self.blocks(images)
        logit = self.linear(convolved.view(images.shape[0], -1))
        return torch.sigmoid(logit)
