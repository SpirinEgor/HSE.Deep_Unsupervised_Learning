from itertools import cycle
from math import ceil

import torch
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import trange

from gan.sn_gan import SNGanGenerator128to32x32, SNGanDiscriminator32x32to128


class WGanGradientPolicy:

    _gp_cf = 10

    def __init__(self, n_filters: int, device: torch.device):
        noise = torch.distributions.Normal(torch.tensor(0.0, device=device), torch.tensor(1.0, device=device))
        self._generator = SNGanGenerator128to32x32(noise, n_filters)
        self._discriminator = SNGanDiscriminator32x32to128(n_filters)
        self._device = device

    def gradient_penalty(self, real_data: Tensor, fake_data: Tensor) -> Tensor:
        batch_size = real_data.shape[0]

        # Calculate interpolation
        eps = torch.rand((batch_size, 1, 1, 1), device=self._device)
        eps = eps.expand_as(real_data)
        interpolated = eps * real_data.data + (1 - eps) * fake_data.data
        interpolated.requires_grad = True

        d_output = self._discriminator(interpolated)
        gradients = torch.autograd.grad(
            outputs=d_output,
            inputs=interpolated,
            grad_outputs=torch.ones(d_output.size(), device=self._device),
            create_graph=True,
            retain_graph=True,
        )[0]

        gradients = gradients.reshape(batch_size, -1)
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        return ((gradients_norm - 1) ** 2).mean()

    def train(
        self,
        train_dataloader: DataLoader,
        n_iterations: int,
        lr: float = 2e-4,
        beta1: float = 0,
        beta2: float = 0.9,
        n_critic_step: int = 5,
    ):
        self._generator.train()
        self._discriminator.train()

        n_epochs = int(ceil(n_critic_step * n_iterations / len(train_dataloader)))

        g_optim = AdamW(self._generator.parameters(), lr=lr, betas=(beta1, beta2))
        g_scheduler = LambdaLR(g_optim, lr_lambda=lambda e: (n_epochs - e) / n_epochs)
        d_optim = AdamW(self._discriminator.parameters(), lr=lr, betas=(beta1, beta2))
        d_scheduler = LambdaLR(d_optim, lr_lambda=lambda e: (n_epochs - e) / n_epochs)

        epoch_bar = trange(n_iterations, desc="Train")
        postfix = {}
        losses = []
        total_iter = 0
        for _ in range(n_epochs):
            for i, batch in enumerate(train_dataloader):
                batch = batch.float().to(self._device)

                # critic update
                d_optim.zero_grad()
                fake_data = self._generator.sample(batch.shape[0])
                gp = self.gradient_penalty(batch, fake_data)
                d_loss = self._discriminator(fake_data).mean() - self._discriminator(batch).mean() + self._gp_cf * gp
                d_loss.backward(retain_graph=True)
                d_optim.step()
                postfix["d_loss"] = d_loss.item()

                # generator update
                if i % n_critic_step == 0:
                    g_optim.zero_grad()
                    fake_data = self._generator.sample(batch.shape[0])
                    g_loss = -self._discriminator(fake_data).mean()
                    g_loss.backward()
                    g_optim.step()
                    losses.append(g_loss.item())
                    postfix["g_loss"] = losses[-1]

                epoch_bar.set_postfix(postfix)
                epoch_bar.update()

                total_iter += 1
                if total_iter == n_iterations:
                    break

            g_scheduler.step()
            d_scheduler.step()
        return losses

    def sample(self, n_samples: int) -> Tensor:
        self._generator.eval()
        with torch.no_grad():
            samples = self._generator.sample(n_samples).permute(0, 2, 3, 1) * 0.5 + 0.5
        return samples.detach().cpu().numpy()
