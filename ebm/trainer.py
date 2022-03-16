from typing import Tuple

import torch
from numpy import ndarray, array
from statsmodels.stats import contrast
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import trange

from ebm.ebm_model import EBM, ConditionalEBM


def train_ebm(
    model: EBM,
    train_dataloader: DataLoader,
    n_epochs: int,
    device: torch.device,
    lr: float = 1e-3,
    alpha: float = 0.1,
    noise: float = 0.005,
    betas=(0.0, 0.999),
) -> Tuple[ndarray, ndarray]:
    optim = AdamW(model.parameters(), lr=lr, betas=betas)

    epoch_bar = trange(n_epochs, desc="Epochs")
    postfix = {}
    con_losses = []
    reg_losses = []
    for _ in epoch_bar:
        for i, batch in enumerate(train_dataloader):
            batch = batch[0].to(device)

            x_real = batch + torch.randn_like(batch) * noise
            x_fake = model.langevin_sample(batch.shape[0], noise=noise)

            loss_real = model(x_real)
            loss_fake = model(x_fake)

            contrastive_loss = loss_fake.mean() - loss_real.mean()
            reg_loss = alpha * (loss_real ** 2 + loss_fake ** 2).mean()

            loss = contrastive_loss + reg_loss
            optim.zero_grad()
            loss.backward()
            optim.step()

            con_losses.append(contrastive_loss.item())
            reg_losses.append(reg_loss.item())

            postfix["batch"] = f"{i + 1}/{len(train_dataloader)}"
            postfix["con_loss"] = con_losses[-1]
            postfix["reg_loss"] = reg_losses[-1]
            epoch_bar.set_postfix(postfix)

    epoch_bar.close()
    return array(con_losses), array(reg_losses)


def train_conditional_ebm(
    model: ConditionalEBM,
    train_dataloader: DataLoader,
    n_epochs: int,
    device: torch.device,
    lr: float = 1e-3,
    alpha: float = 0.1,
    noise: float = 0.005,
    betas=(0.0, 0.999),
) -> ndarray:
    optim = AdamW(model.parameters(), lr=lr, betas=betas)

    epoch_bar = trange(n_epochs, desc="Epochs")
    postfix = {}
    losses = []
    for _ in epoch_bar:
        for i, batch in enumerate(train_dataloader):
            x_real = batch[0].to(device)
            labels = batch[1].to(device)

            loss_clf = F.cross_entropy(model(x_real), labels)

            x_fake = model.langevin_sample(x_real.shape[0], noise=noise)

            loss_real = torch.logsumexp(model(x_real), dim=1)
            loss_fake = torch.logsumexp(model(x_fake), dim=1)

            contrastive_loss = loss_fake.mean() - loss_real.mean()
            reg_loss = alpha * (loss_real ** 2 + loss_fake ** 2).mean()

            loss = contrastive_loss + loss_clf + reg_loss

            optim.zero_grad()
            loss.backward()
            optim.step()

            losses.append(loss.item())

            postfix["batch"] = f"{i + 1}/{len(train_dataloader)}"
            postfix["loss"] = losses[-1]
            epoch_bar.set_postfix(postfix)

    epoch_bar.close()
    return array(losses)
