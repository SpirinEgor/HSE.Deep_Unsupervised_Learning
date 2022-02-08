from itertools import chain

import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm, trange

from context_encoder.modules import MaskedImageEncoder, PatchDecoder, PatchDiscriminator


class ContextEncoder:
    def __init__(self, latent_dim: int, device: torch.device):
        self.encoder = MaskedImageEncoder(latent_dim).to(device)
        self.decoder = PatchDecoder(latent_dim).to(device)
        self.discriminator = PatchDiscriminator().to(device)
        self.device = device

    def to_mnist(self, patch):
        mask = patch < 0
        patch[mask] = -1
        patch[~mask] = 1
        return patch

    def _train_epoch(self, train_dataloader: DataLoader, ed_optim: Optimizer, d_optim: Optimizer):
        self.encoder.train()
        self.decoder.train()
        self.discriminator.train()

        rec_losses = []
        adv_losses = []
        postfix = {}
        data_pbar = tqdm(train_dataloader, leave=False, desc="Training", postfix=postfix)
        for images, patches in data_pbar:
            images = images.to(self.device)
            patches = patches.to(self.device)
            bs = images.shape[0]

            with torch.no_grad():
                embeddings = self.encoder(images)
                reconstruction = self.decoder(embeddings)
                reconstruction = self.to_mnist(reconstruction)

            real_patches = self.discriminator(patches)
            fake_patches = self.discriminator(reconstruction)

            d_loss = F.binary_cross_entropy(real_patches, torch.ones((bs, 1), device=self.device))
            d_loss += F.binary_cross_entropy(fake_patches, torch.zeros((bs, 1), device=self.device))

            d_optim.zero_grad()
            d_loss.backward()
            d_optim.step()

            embeddings = self.encoder(images)
            reconstruction = self.decoder(embeddings)
            mse_loss = F.mse_loss(reconstruction, patches)

            with torch.no_grad():
                discriminator_ans = self.discriminator(reconstruction)
            adv_loss = F.binary_cross_entropy(discriminator_ans, torch.ones((bs, 1), device=self.device))

            loss = adv_loss + mse_loss

            ed_optim.zero_grad()
            loss.backward()
            ed_optim.step()

            adv_losses.append(adv_loss.item())
            rec_losses.append(mse_loss.item())
            postfix = {"mse": rec_losses[-1], "adv": adv_losses[-1]}
            data_pbar.set_postfix(postfix)
        data_pbar.close()

        return rec_losses, adv_losses

    def train(self, train_dataloader: DataLoader, n_epochs: int, lr: float):
        ed_optim = torch.optim.AdamW(chain(self.encoder.parameters(), self.decoder.parameters()), lr=lr)
        d_optim = torch.optim.AdamW(self.discriminator.parameters(), lr=lr)

        rec_losses = []
        adv_losses = []

        epoch_bar = trange(n_epochs, desc="Epochs")
        for _ in epoch_bar:
            epoch_rec_loss, epoch_adv_losses = self._train_epoch(train_dataloader, ed_optim, d_optim)
            rec_losses += epoch_rec_loss
            adv_losses += epoch_adv_losses
        epoch_bar.close()
        return rec_losses, adv_losses

    def eval(self):
        self.encoder.eval()
        self.decoder.eval()
        self.discriminator.eval()
