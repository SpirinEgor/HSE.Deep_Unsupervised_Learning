from typing import List, Tuple

import torch
from torch import Tensor
from torch import nn
from torch.distributions import MultivariateNormal
from torchdiffeq import odeint

DOUBLE_TENSOR = Tuple[Tensor, Tensor]
TRIPLE_TENSOR = Tuple[Tensor, Tensor, Tensor]


class ContiguousNormalizedFlow(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], width: int, base_dist_args: List[Tensor]):
        super().__init__()
        self._input_dim = input_dim
        self._width = width
        self._block_dim = input_dim * width

        layers = [nn.Linear(1, hidden_dims[0]), nn.Tanh()]
        for i in range(1, len(hidden_dims)):
            layers.extend([nn.Linear(hidden_dims[i - 1], hidden_dims[i]), nn.Tanh()])

        self._model = nn.Sequential(*layers)
        self._u_model = nn.Linear(hidden_dims[-1], self._block_dim)
        self._w_model = nn.Linear(hidden_dims[-1], self._block_dim)
        self._b_model = nn.Linear(hidden_dims[-1], self._width)

        self._base_dist = MultivariateNormal(*base_dist_args)

    def _get_u_w_b(self, t: Tensor) -> TRIPLE_TENSOR:
        model_out = self._model(t.unsqueeze(0)).squeeze(0)
        u = self._u_model(model_out).reshape(self._width, 1, self._input_dim)
        w = self._w_model(model_out).reshape(self._width, self._input_dim, 1)
        b = self._b_model(model_out).reshape(self._width, 1, 1)
        return u, w, b

    def _get_dz_dt(self, z: Tensor, t: Tensor) -> Tensor:
        u, w, b = self._get_u_w_b(t)
        z_r = z.unsqueeze(0).repeat(self._width, 1, 1)
        lin = z_r.matmul(w) + b
        h = torch.tanh(lin).matmul(u)
        return h.mean(dim=0)

    def _get_dlog_p_dt(self, f: Tensor, z: Tensor) -> Tensor:
        batch_size = z.shape[0]
        m_trace = z.new_zeros(batch_size, dtype=torch.float32)
        for i in range(self._input_dim):
            m_trace -= torch.autograd.grad(f[:, i].sum(), z, create_graph=True)[0][:, i]
        return m_trace.reshape(batch_size, 1)

    @torch.enable_grad()
    def forward(self, t: Tensor, ode_input: DOUBLE_TENSOR) -> DOUBLE_TENSOR:
        z = ode_input[0]
        z.requires_grad_(True)
        dz_dt = self._get_dz_dt(z, t)
        dlog_p_dt = self._get_dlog_p_dt(dz_dt, z)
        return dz_dt, dlog_p_dt

    def _flow(self, z_1: Tensor, t_0: float, t_1: float, tolerance: float = 1e-5) -> DOUBLE_TENSOR:
        bs = z_1.shape[0]
        dlog_p_dt_1 = z_1.new_zeros((bs, 1), dtype=torch.float32)
        time_interval = z_1.new_tensor([t_1, t_0], dtype=torch.float32)

        z_t, m_log_det_t = odeint(
            self, (z_1, dlog_p_dt_1), time_interval, atol=tolerance, rtol=tolerance, method="dopri5"
        )

        return z_t[-1], -m_log_det_t[-1]

    def log_prob(self, batch: Tensor, t_0: float, t_1: float, tolerance: float = 1e-5) -> Tensor:
        bs = batch.shape[0]
        z, log_det = self._flow(batch, t_0, t_1, tolerance)
        lop_p_z = self._base_dist.log_prob(z).reshape(bs, 1)
        return lop_p_z + log_det

    @torch.no_grad()
    def calc_probability(self, batch: Tensor, t_0: float, t_1: float, tolerance: float = 1e-5) -> Tensor:
        return self.log_prob(batch, t_0, t_1, tolerance).exp()

    @torch.no_grad()
    def extract_latent_vector(self, batch: Tensor, t_0: float, t_1: float, tolerance: float) -> Tensor:
        return self._flow(batch, t_0, t_1, tolerance)[0]


class HutchinsonCNF(ContiguousNormalizedFlow):
    def _get_dlog_p_dt(self, f: Tensor, z: Tensor) -> Tensor:
        v = z.new_empty((self._input_dim, 1))
        torch.randint(0, 2, v.shape, out=v)

        a = torch.autograd.grad(f.matmul(v).sum(), z, create_graph=True)[0]
        return -a.matmul(v)
