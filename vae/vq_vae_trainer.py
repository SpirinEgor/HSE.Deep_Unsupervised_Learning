from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from numpy import ndarray, array
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import trange

from vae.vq_vae_impl import VqVae


class VqVaeTrainer:
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device,
        beta: float = 1.0,
    ):
        self._vq_vae = VqVae(in_channels, hidden_dim, num_embeddings, embedding_dim)
        self._vq_vae = self._vq_vae.to(device)

        self._beta = beta
        self._device = device

    def _calculate_loss(self, images: Tensor, reconstructed: Tensor, emb: Tensor, quantized_emb: Tensor) -> Tensor:
        recon_loss = F.mse_loss(images, reconstructed)
        vq_loss = F.mse_loss(emb.detach(), quantized_emb)
        commitment_loss = F.mse_loss(emb, quantized_emb.detach())

        return recon_loss + vq_loss + self._beta * commitment_loss

    def train(
        self,
        train_dataloader: DataLoader,
        n_epochs: int,
        lr: float = 0.001,
        test_dataloader: Optional[DataLoader] = None,
    ) -> Tuple[ndarray, ...]:
        optimizer = torch.optim.AdamW(self._vq_vae.parameters(), lr=lr)

        train_losses, test_losses = [], []

        test_losses.append(self.test(test_dataloader))

        epoch_bar = trange(n_epochs, desc="Training")
        postfix = {"test_loss": test_losses[-1]}
        for _ in epoch_bar:
            self._vq_vae.train()
            for batch in train_dataloader:
                batch = batch.to(self._device)

                reconstructed, emb, quantized_emb = self._vq_vae(batch)
                loss = self._calculate_loss(batch, reconstructed, emb, quantized_emb)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())
                postfix["train_loss"] = loss.item()
                epoch_bar.set_postfix(postfix)

            if test_dataloader is None:
                continue
            test_losses.append(self.test(test_dataloader))
            postfix["test_loss"] = test_losses[-1]
            epoch_bar.set_postfix(postfix)

        epoch_bar.close()

        return (array(train_losses), array(test_losses)) if test_dataloader is not None else array(train_losses)

    def test(self, test_dataloader: DataLoader) -> float:
        self._vq_vae.eval()
        loss = 0
        for batch in test_dataloader:
            with torch.no_grad():
                batch = batch.to(self._device)

                reconstructed, emb, quantized_emb = self._vq_vae(batch)
                cur_loss = self._calculate_loss(batch, reconstructed, emb, quantized_emb)
                loss += cur_loss.item()
        n = len(test_dataloader)
        return loss / n

    @property
    def vq_vae(self) -> VqVae:
        return self._vq_vae
