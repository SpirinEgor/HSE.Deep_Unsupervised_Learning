from typing import Tuple

import numpy as np
import torch
from torch.nn import functional as F
from torch.distributions import Normal, Uniform
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm, trange

from realnvp.modules import RealNVP


class RealNVPTrainer:
    def __init__(self, device: torch.device, alpha: float = 0.05, hidden_dim: int = 128):
        self.real_nvp = RealNVP(hidden_dim).to(device)
        self.device = device
        self.bdist = Normal(torch.tensor([0.0], device=device), torch.tensor([1.0], device=device))
        self.alpha = alpha

    def log_prob(self, batch: torch.Tensor) -> torch.Tensor:
        z, log_det = self.real_nvp(batch)
        p_x = log_det.sum(dim=(1, 2, 3))
        p_z = self.bdist.log_prob(z).sum(dim=(1, 2, 3))
        return p_x + p_z

    def preprocess(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch += Uniform(0, 0.1).sample(batch.shape).to(self.device)
        batch = batch.clamp(0, 1)

        batch = batch * (1 - self.alpha) + self.alpha

        # NaNs arise so some regularization of log is required
        # last term is from normalization
        logit = torch.log(batch + 1e-8) - torch.log(1 - batch + 1e-8) + np.log(1 - self.alpha) - np.log(3)
        log_det = F.softplus(logit) + F.softplus(-logit)

        return logit, log_det.sum(dim=(1, 2, 3))

    def common_step(self, batch: torch.Tensor) -> torch.Tensor:
        x, log_det = self.preprocess(batch)
        log_prob = self.log_prob(x)
        log_prob += log_det
        return -log_prob.mean() / (3 * 32 * 32)

    @torch.no_grad()
    def test(self, test_dataloader: DataLoader) -> np.ndarray:
        self.real_nvp.eval()
        losses = []
        for batch in tqdm(test_dataloader, desc="Testing", leave=False):
            batch = batch.to(self.device)
            losses.append(self.common_step(batch).item())
        return sum(losses) / len(losses)

    def fit(self, train_dataloader: DataLoader, test_dataloader: DataLoader, epochs: int = 10, lr: float = 1e-4):
        train_losses = []
        test_losses = [self.test(test_dataloader)]

        optim = AdamW(self.real_nvp.parameters(), lr=lr)

        pbar = trange(epochs, desc="Training")
        for _ in pbar:
            self.real_nvp.train()
            for i, batch in enumerate(train_dataloader):
                batch = batch.to(self.device)
                loss = self.common_step(batch)

                optim.zero_grad()
                loss.backward()
                optim.step()

                train_losses.append(loss.item())
                pbar.set_postfix({"loss": train_losses[-1], "batch": f"{i + 1}/{len(train_dataloader)}"})

            test_losses.append(self.test(test_dataloader))
            tqdm.write(f"Test loss: {test_losses[-1]}")
        pbar.close()

        return train_losses, test_losses

    @torch.no_grad()
    def sample2img(self, x: torch.Tensor) -> np.ndarray:
        x = 1 / (1 + torch.exp(-x))
        x = x - self.alpha
        x = x / (1 - self.alpha)
        x = x.permute(0, 2, 3, 1).clamp(0, 1)
        return x.cpu().numpy()

    @torch.no_grad()
    def sample(self, n: int):
        self.real_nvp.eval()
        z = self.bdist.sample((n, 3, 32, 32)).squeeze(-1)
        return self.sample2img(self.real_nvp.generate(z))

    @torch.no_grad()
    def interpolate(self, images: torch.Tensor) -> np.ndarray:
        self.real_nvp.eval()

        images = images.to(self.device).float()
        assert images.shape[0] % 2 == 0
        start_size = images.shape[0] // 2

        x, _ = self.preprocess(images)
        z, _ = self.real_nvp(x)

        latents = []
        for i in range(0, start_size):
            z_start = z[i].unsqueeze(0)
            z_finish = z[start_size + i].unsqueeze(0)

            d = (z_finish - z_start) / 5

            latents.append(z_start)
            for j in range(1, 5):
                latents.append(z_start + d * j)
            latents.append(z_finish)

        latents = torch.cat(latents)
        res = self.real_nvp.generate(latents)
        return self.sample2img(res)
