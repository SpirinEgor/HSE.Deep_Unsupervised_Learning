from itertools import chain

import torch
from numpy import ndarray
from torch import nn, Tensor
from torch.distributions import Normal
from torch.nn.functional import cross_entropy
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import trange

from gan.bi_gan_modules import BiGanGenerator, BiGanDiscriminator, BiGanEncoder


class BiGAN:
    def __init__(
        self, img_sz: int, latent_dim: int, device: torch.device, hidden_dim: int = 1024, leaky_relu_cf: float = 0.2
    ):
        self._img_sz = img_sz
        self._latent_dim = latent_dim
        self._device = device

        self._generator = BiGanGenerator(img_sz, latent_dim, hidden_dim).to(device)
        self._discriminator = BiGanDiscriminator(latent_dim, img_sz * img_sz, hidden_dim, leaky_relu_cf).to(device)
        self._encoder = BiGanEncoder(img_sz * img_sz, latent_dim, hidden_dim, leaky_relu_cf).to(device)

        self._classifier = nn.Linear(latent_dim, 10).to(device)

        self._base_dist = Normal(
            torch.tensor(0, dtype=torch.float32, device=device), torch.tensor(1, dtype=torch.float32, device=device)
        )

    def fit_classifier(
        self, train_dataloader: DataLoader, test_dataloader: DataLoader, n_epochs: int, lr: float = 1e-3
    ):
        torch.nn.init.xavier_uniform_(self._classifier.weight)
        torch.nn.init.zeros_(self._classifier.bias)
        optim = AdamW(self._classifier.parameters(), lr=lr)

        train_losses, test_losses = [], []
        epoch_bar = trange(n_epochs, desc="Classifier training")
        postfix = {}
        for _ in epoch_bar:
            self._classifier.train()
            for i, (batch, labels) in enumerate(train_dataloader):
                batch = batch.to(self._device)
                labels = labels.to(self._device)
                optim.zero_grad()

                with torch.no_grad():
                    z = self._encoder(batch)

                loss = cross_entropy(self._classifier(z), labels)
                loss.backward()
                optim.step()

                train_losses.append(loss.item())
                postfix["batch"] = f"{i + 1}/{len(train_dataloader)}"
                postfix["train loss"] = train_losses[-1]
                epoch_bar.set_postfix(postfix)

            self._classifier.eval()
            epoch_test_loss = 0
            for batch, labels in test_dataloader:
                batch = batch.to(self._device)
                labels = labels.to(self._device)
                with torch.no_grad():
                    z = self._encoder(batch)
                loss = cross_entropy(self._classifier(z), labels)
                epoch_test_loss += loss.item()
            test_losses.append(epoch_test_loss / len(test_dataloader))
            postfix["test loss"] = test_losses[-1]
            epoch_bar.set_postfix(postfix)

        epoch_bar.close()
        return train_losses, test_losses

    def gan_step(self, real: Tensor) -> Tensor:
        z_real = self._encoder(real)
        real_score = self._discriminator(z_real, real)

        z_fake = self._base_dist.sample(z_real.shape)
        fake = self._generator(z_fake)
        fake_score = self._discriminator(z_fake, fake)

        loss = torch.log(real_score) + torch.log(1 - fake_score)
        return -loss.mean()

    def fit_bi_gan(self, train_dataloader: DataLoader, test_dataloader: DataLoader, n_epochs: int, lr: float = 1e-4):
        g_optim = AdamW(chain(self._generator.parameters(), self._encoder.parameters()), lr=lr)
        g_scheduler = LambdaLR(g_optim, lambda epoch: (n_epochs - epoch) / n_epochs)

        d_optim = AdamW(self._discriminator.parameters(), lr=lr)
        d_scheduler = LambdaLR(d_optim, lambda epoch: (n_epochs - epoch) / n_epochs)

        train_losses, test_losses = [], []
        epoch_bar = trange(n_epochs, desc="BiGAN training")
        postfix = {}
        for _ in epoch_bar:
            self._generator.train()
            self._discriminator.train()
            self._encoder.train()
            for i, (batch, _) in enumerate(train_dataloader):
                batch = batch.to(self._device)

                d_optim.zero_grad()
                d_loss = self.gan_step(batch)
                d_loss.backward()
                d_optim.step()

                g_optim.zero_grad()
                g_loss = -self.gan_step(batch)
                g_loss.backward()
                g_optim.step()

                train_losses.append(d_loss.item())
                postfix["batch"] = f"{i + 1}/{len(train_dataloader)}"
                postfix["train loss"] = train_losses[-1]
                epoch_bar.set_postfix(postfix)

            self._generator.eval()
            self._discriminator.eval()
            self._encoder.eval()
            epoch_test_loss = 0
            for batch, labels in test_dataloader:
                batch = batch.to(self._device)
                labels = labels.to(self._device)
                z = self._encoder(batch)
                loss = cross_entropy(self._classifier(z), labels)
                epoch_test_loss += loss.item()
            test_losses.append(epoch_test_loss / len(test_dataloader))
            postfix["test classifier loss"] = test_losses[-1]

            d_scheduler.step()
            g_scheduler.step()

        epoch_bar.close()
        return train_losses, test_losses

    @torch.no_grad()
    def _tensor2image(self, tensor: Tensor) -> ndarray:
        tensor = 0.5 * tensor + 0.5
        images = tensor.detach().cpu().numpy().transpose(0, 2, 3, 1)
        return images

    @torch.no_grad()
    def sample(self, n: int) -> ndarray:
        z = 2 * (torch.rand(n, self._latent_dim).to(self._device) - 0.5)
        self._generator.eval()
        samples = self._generator(z)
        return self._tensor2image(samples)

    @torch.no_grad()
    def reconstruct(self, batch: Tensor) -> ndarray:
        batch = batch.to(self._device)
        self._encoder.eval()
        self._generator.eval()
        z = self._encoder(batch)
        batch_reconstructed = self._generator(z)
        pairs = torch.cat((batch, batch_reconstructed))
        return self._tensor2image(pairs)
