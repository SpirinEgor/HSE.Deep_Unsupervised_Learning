from typing import Tuple

import torch
from torch import nn
from torch.distributions import ExponentialFamily

from made.made_model import MADE


class FlowMade(nn.Module):
    def __init__(self, n_features: int, made_hidden_layers, base_dist: ExponentialFamily):
        super().__init__()
        self._n_features = n_features
        # From `n features` to `2 * n features`
        self._made = MADE(n_features, 2, made_hidden_layers, use_one_hot=False)
        self._base_dist = base_dist

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param z: tensor of shape [bs; n features]
        """
        # [bs; 2; n features]
        made_out = self._made(z)
        # [bs; 1; n_features]
        mu, log_sigma = torch.chunk(made_out, 2, dim=1)
        return mu.squeeze(1), log_sigma.squeeze(1)

    def flow(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, log_sigma = self(z)
        return mu + z * log_sigma.exp(), log_sigma

    def inverse_flow(self, e: torch.Tensor, mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
        return (e - mu) * (-log_sigma).exp()

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        e, log_det = self.flow(z)

        return self._base_dist.log_prob(e) + log_det
