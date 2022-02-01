from typing import Tuple

import torch
from numpy import ndarray, array
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import trange

from ebm.ebm_model import EBM


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
