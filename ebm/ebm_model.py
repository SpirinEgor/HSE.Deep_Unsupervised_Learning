from contextlib import contextmanager

import torch
from numpy import sqrt
from torch import nn

from ebm.buffer import Buffer


@contextmanager
def no_grad_mode(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    try:
        yield model
    finally:
        for p in model.parameters():
            p.requires_grad = True
        model.train()


class EBM(nn.Module):
    def __init__(self, buffer: Buffer):
        super().__init__()
        self._buffer = buffer
        self._net = nn.Sequential(
            nn.Conv2d(1, 16, 5, 2, 4),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.SiLU(),
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self._net(batch)

    def langevin_sample(self, size: int, n_iter: int = 60, eps_init: float = 10.0, noise: float = 0.005):
        with no_grad_mode(self):
            from_buffer = int(size * 0.95)
            x0_buffer = self._buffer.sample(from_buffer)
            x0_noise = x0_buffer.new_empty((size - from_buffer, *x0_buffer.shape[1:])).uniform_(-1.0, 1.0)

            x = torch.vstack((x0_buffer, x0_noise))
            x.requires_grad = True

            for i in range(n_iter):
                eps = eps_init - eps_init * i / n_iter
                z = torch.randn_like(x) * noise
                grad_x = torch.autograd.grad(self(x).sum(), x)[0].clamp(-0.03, 0.03)
                x = torch.clip(x + sqrt(2 * eps) * z + eps * grad_x, -1.0, 1.0)

            self._buffer.push(x.detach())

        return x
