from itertools import chain
from typing import Optional, Tuple, List

import torch
import torch.nn.functional as F
from numpy import array, ndarray
from torch.distributions import Normal
from torch.utils.data import DataLoader
from tqdm.auto import trange

from flow.flow_made import FlowMade
from vae import Vae32x32


class VAETrainer:
    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        device: torch.device,
        use_prior: bool = False,
        made_hidden_dims: List[int] = None,
    ):
        self._use_prior = use_prior
        self._latent_dim = latent_dim
        self._device = device
        self._vae = Vae32x32(in_channels, latent_dim).to(device)

        if use_prior:
            base_dist = Normal(
                torch.tensor(0, dtype=torch.float, device=device),
                torch.tensor(1, dtype=torch.float, device=device),
            )
            self._flow_made = FlowMade(latent_dim, made_hidden_dims, base_dist).to(device)

    @property
    def vae_model(self) -> Vae32x32:
        return self._vae

    @staticmethod
    def _calc_reconstruction_loss(original_images: torch.Tensor, reconstructed_images: torch.Tensor) -> torch.Tensor:
        batch_size = original_images.shape[0]
        element_wise = F.mse_loss(original_images, reconstructed_images, reduction="none")
        return element_wise.reshape(batch_size, -1).sum(dim=1).mean()

    def _calc_kl_loss(self, mu: torch.Tensor, log_sigma: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if self._use_prior:
            dist = Normal(mu, log_sigma.exp())
            log_q_z_x = dist.log_prob(z).flatten(1)
            log_p_z = self._flow_made.log_prob(z.flatten(1))
            return (log_q_z_x - log_p_z).sum(dim=1).mean()
        else:
            return (0.5 * (torch.exp(2 * log_sigma) + mu ** 2 - 2 * log_sigma - 1)).sum(dim=1).mean()

    def _calc_loss_from_batch(self, batch: torch.Tensor, beta: float) -> Tuple[torch.Tensor, ...]:
        reconstruction, (z, mu, log_sigma) = self._vae(batch)
        r_loss_val = self._calc_reconstruction_loss(batch.detach(), reconstruction)
        kl_loss_val = self._calc_kl_loss(mu, log_sigma, z)
        total_loss = r_loss_val + beta * kl_loss_val
        return total_loss, r_loss_val, kl_loss_val

    def train(
        self,
        train_dataloader: DataLoader,
        n_epochs: int,
        lr: float = 0.001,
        test_dataloader: Optional[DataLoader] = None,
        beta: float = 1.0,
    ) -> Tuple[ndarray, ...]:
        parameters = self._vae.parameters()
        if self._use_prior:
            parameters = chain(parameters, self._flow_made.parameters())
        optimizer = torch.optim.AdamW(parameters, lr=lr)

        train_losses, test_losses = [], []

        test_losses.append(self.test(test_dataloader, beta))

        epoch_bar = trange(n_epochs, desc="Training")
        postfix = {"test_elbo": test_losses[-1][0]}
        for _ in epoch_bar:
            self._vae.train()
            for batch in train_dataloader:
                batch = batch.to(self._device)
                total_loss, r_loss_val, kl_loss_val = self._calc_loss_from_batch(batch, beta)

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                train_losses.append([total_loss.item(), r_loss_val.item(), kl_loss_val.item()])
                postfix["train_elbo"] = total_loss.item()
                epoch_bar.set_postfix(postfix)

            if train_dataloader is None:
                continue
            test_losses.append(self.test(test_dataloader, beta))
            postfix["test_elbo"] = test_losses[-1][0]
            epoch_bar.set_postfix(postfix)

        epoch_bar.close()

        return (array(train_losses), array(test_losses)) if test_dataloader is not None else array(train_losses)

    def test(self, test_dataloader: DataLoader, beta: float = 1.0) -> Tuple[float, ...]:
        self._vae.eval()
        t_loss, r_loss, kl_loss = 0, 0, 0
        for batch in test_dataloader:
            with torch.no_grad():
                batch = batch.to(self._device)
                total_loss, r_loss_val, kl_loss_val = self._calc_loss_from_batch(batch, beta)

                r_loss += r_loss_val.item()
                kl_loss += kl_loss_val.item()
                t_loss += total_loss.item()
        n = len(test_dataloader)
        return t_loss / n, r_loss / n, kl_loss / n

    @torch.no_grad()
    def sample(self, n: int) -> torch.Tensor:
        z = torch.randn(n, self._latent_dim, device=self._device)

        if self._use_prior:
            for i in range(self._latent_dim):
                mu, log_sigma = self._flow_made(z)
                z[:, i] = self._flow_made.inverse_flow(z[:, i], mu[:, i], log_sigma[:, i])

        z = z.unsqueeze(-1).unsqueeze(-1)
        samples = torch.clip(self._vae.decode(z), -1, 1)  # 0, 1
        return samples.permute(0, 2, 3, 1)
