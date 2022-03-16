from contextlib import contextmanager

import torch
from numpy import sqrt
from torch import nn

from ebm.buffer import Buffer


def _prepare_start_x(buffer: Buffer, size: int):
    from_buffer = int(size * 0.95)
    x0_buffer = buffer.sample(from_buffer)
    x0_noise = x0_buffer.new_empty((size - from_buffer, *x0_buffer.shape[1:])).uniform_(-1.0, 1.0)

    x = torch.vstack((x0_buffer, x0_noise))
    x.requires_grad = True
    return x


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
            x = _prepare_start_x(self._buffer, size)

            for i in range(n_iter):
                eps = eps_init - eps_init * i / n_iter
                z = torch.randn_like(x) * noise
                grad_x = torch.autograd.grad(self(x).sum(), x)[0].clamp(-0.03, 0.03)
                x = torch.clamp(x + sqrt(2 * eps) * z + eps * grad_x, -1.0, 1.0)

            self._buffer.push(x.detach())

        return x


class ConditionalEBM(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, buffer: Buffer, hidden_dim: int = 128, n_layers: int = 4):
        super().__init__()
        self._buffer = buffer
        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, num_classes))
        self._net = nn.Sequential(*layers)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self._net(batch)

    def langevin_sample(
        self, size: int, class_num: int = None, n_iter: int = 500, eps_init: float = 0.1, noise: float = 0.005
    ):
        with no_grad_mode(self):
            x = _prepare_start_x(self._buffer, size)

            for i in range(n_iter):
                eps = eps_init - eps_init * i / n_iter
                z = torch.randn_like(x) * noise

                if class_num is None:
                    grad_x = torch.autograd.grad(torch.logsumexp(self(x), dim=1).sum(), x)[0]
                else:
                    grad_x = torch.autograd.grad(self(x)[:, class_num].sum(), x)[0]
                # grad_x = grad_x.clamp(-0.03, 0.03)

                # Smile border
                x = torch.clamp(x + sqrt(2 * eps) * z + eps * grad_x, -3, 3)

            self._buffer.push(x.detach())

        return x
