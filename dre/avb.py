import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm, trange

from dre.modules import Encoder, Decoder, Classifier


class AVB:
    def __init__(self, latent_dim: int, device: torch.device, noise_dim: int = 16):
        super().__init__()

        self.encoder = Encoder(latent_dim=latent_dim, noise_dim=noise_dim).to(device)
        self.decoder = Decoder(latent_dim=latent_dim).to(device)
        self.clf = Classifier(latent_dim=latent_dim).to(device)

        self.noise_dist = MultivariateNormal(torch.zeros(noise_dim, device=device), torch.eye(noise_dim, device=device))
        self.z_dist = MultivariateNormal(torch.zeros(latent_dim, device=device), torch.eye(latent_dim, device=device))

        self.device = device

    def _gen_loss(self, batch: torch.Tensor) -> torch.Tensor:
        noise = self.noise_dist.sample((batch.shape[0],)).to(self.device)
        z_encoded = self.encoder(batch, noise)

        T = self.clf(batch, z_encoded).mean()
        recon = self.decoder(z_encoded)
        loss = T + F.binary_cross_entropy_with_logits(recon, batch, reduction="sum") / batch.shape[0]
        return loss

    def _clf_loss(self, batch):
        z = self.z_dist.sample((batch.shape[0],)).to(self.device)
        noise = self.noise_dist.sample((batch.shape[0],)).to(self.device)
        z_encoded = self.encoder(batch, noise)

        T_real = self.clf(batch, z_encoded)
        T_fake = self.clf(batch, z)
        real_labels = torch.ones_like(T_real)
        fake_labels = torch.zeros_like(T_fake)
        loss_real = F.binary_cross_entropy_with_logits(T_real, real_labels)
        loss_fake = F.binary_cross_entropy_with_logits(T_fake, fake_labels)
        return loss_real + loss_fake

    @torch.no_grad()
    def _test(self, test_loader: DataLoader):
        for m in [self.encoder, self.decoder, self.clf]:
            m.eval()

        elbo_losses = 0
        clf_losses = 0

        for batch in tqdm(test_loader, desc="Testing", leave=False):
            batch = batch.to(self.device)
            elbo_loss = self._gen_loss(batch)
            clf_loss = self._clf_loss(batch)
            elbo_losses += elbo_loss.item()
            clf_losses += clf_loss.item()

        return elbo_losses / len(test_loader), clf_losses / len(test_loader)

    def fit(self, train_loader: DataLoader, test_loader: DataLoader, n_epochs: int = 20, lr: float = 1e-3):
        train_losses = []
        test_losses = []

        decoder_optim = AdamW(self.decoder.parameters(), lr=lr)
        encoder_optim = AdamW(self.encoder.parameters(), lr=lr)
        clf_optim = AdamW(self.clf.parameters(), lr=lr)

        test_losses.append(self._test(test_loader))

        for _ in trange(n_epochs, desc="Epochs"):
            for m in [self.encoder, self.decoder, self.clf]:
                m.train()

            train_tqdm = tqdm(train_loader, desc="Training", leave=False)
            postfix = {}
            for batch in train_tqdm:
                batch = batch.to(self.device)

                elbo_loss = self._gen_loss(batch)

                encoder_optim.zero_grad()
                decoder_optim.zero_grad()
                elbo_loss.backward()
                encoder_optim.step()
                decoder_optim.step()

                clf_loss = self._clf_loss(batch)
                clf_optim.zero_grad()
                clf_loss.backward()
                clf_optim.step()

                train_losses.append((elbo_loss.item(), clf_loss.item()))
                postfix["elbo loss"] = train_losses[-1][0]
                postfix["clf loss"] = train_losses[-1][1]
                train_tqdm.set_postfix(postfix)

            test_losses.append(self._test(test_loader))

        return np.array(train_losses), np.array(test_losses)

    @torch.no_grad()
    def _tensor2image(self, tensor):
        return tensor.clip(0, 1).detach().cpu().numpy()

    @torch.no_grad()
    def sample(self, n):
        z = self.z_dist.sample((n,)).to(self.device)
        return self._tensor2image(self.decoder(z))
