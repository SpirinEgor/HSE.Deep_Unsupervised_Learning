from typing import List, Optional, Tuple

import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm.auto import trange, tqdm

from cnf import ContiguousNormalizedFlow


class CNFTrainer:
    def __init__(
        self,
        cnf_model: ContiguousNormalizedFlow,
        device: torch.device = torch.device("cpu"),
        t_0: float = 0.0,
        t_1: float = 10.0,
        tolerance: float = 1e-5,
    ):
        self._CNF = cnf_model
        self._t_0 = t_0
        self._t_1 = t_1
        self._device = device
        self._tol = tolerance

    def _calculate_loss(self, batch: Tensor) -> Tensor:
        return -self._CNF.log_prob(batch, self._t_0, self._t_1, self._tol).mean()

    def fit(
        self,
        train_dataloader: DataLoader,
        n_epochs: int,
        lr: float = 1e-3,
        weight_decay: float = 0,
        clip_norm: Optional[float] = None,
        test_dataloader: Optional[DataLoader] = None,
    ) -> Tuple[List[float], ...]:
        optimizer = torch.optim.AdamW(self._CNF.parameters(), lr=lr, weight_decay=weight_decay)

        train_losses, test_losses = [], []
        epoch_range = trange(n_epochs, desc="Epochs")
        for _ in epoch_range:
            self._CNF.train()
            epoch_train_losses = []
            for batch in train_dataloader:
                batch = batch.to(self._device)
                loss = self._calculate_loss(batch)
                optimizer.zero_grad()
                loss.backward()
                if clip_norm is not None:
                    clip_grad_norm_(self._CNF.parameters(), clip_norm)
                optimizer.step()

                epoch_train_losses.append(loss.item())

            epoch_loss = sum(epoch_train_losses) / len(epoch_train_losses)
            train_losses.extend(epoch_train_losses)

            if test_dataloader is not None:
                self._CNF.eval()
                epoch_test_losses = []
                for batch in test_dataloader:
                    batch = batch.to(self._device)
                    loss = self._calculate_loss(batch)
                    epoch_test_losses.append(loss.item())
                test_loss = sum(epoch_test_losses) / len(epoch_test_losses)
                test_losses.append(test_loss)
                epoch_range.set_postfix({"train loss": epoch_loss, "test loss": test_loss})
            else:
                epoch_range.set_postfix({"train loss": epoch_loss})
        epoch_range.close()

        return (train_losses, test_losses) if test_dataloader is not None else train_losses

    def calc_probabilities(self, dataloader: DataLoader) -> Tensor:
        return torch.cat(
            [
                self._CNF.calc_probability(batch.to(self._device), self._t_0, self._t_1, self._tol)
                for batch in tqdm(dataloader, desc="Calculating probabilities")
            ]
        )

    def extract_latent_vectors(self, dataloader: DataLoader) -> Tensor:
        return torch.cat(
            [
                self._CNF.extract_latent_vector(batch.to(self._device), self._t_0, self._t_1, self._tol)
                for batch in tqdm(dataloader, desc="Extracting latent vectors")
            ]
        )
