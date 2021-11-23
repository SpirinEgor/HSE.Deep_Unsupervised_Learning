from typing import Dict, Any, Tuple

import torch
from scipy.optimize import bisect
from torch import nn
from torch.distributions import Uniform, Normal

from pixel_cnn.pixel_cnn import PixelCNN


class FlowPixelCNN(nn.Module):
    def __init__(self, mixture_components: int, base_dist: str, pixel_cnn_kwargs: Dict[str, Any]):
        super().__init__()
        self._n_comp = mixture_components
        output = 3 * mixture_components
        self.__pixel_cnn = PixelCNN(out_channels=output, **pixel_cnn_kwargs)

        if base_dist == "uniform":
            self._base_dist = Uniform(0, 1)
        else:
            raise ValueError(f"Unsupported base distribution: {base_dist}")

        self.__ln_2 = nn.Parameter(torch.tensor(2).log())

    def forward(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # [N; 3 * comp; C; H; W]
        pixel_out = self.__pixel_cnn(batch)

        # [N; comp; C; H; W]
        w, mu, log_s = pixel_out.chunk(3, dim=1)
        w = torch.softmax(w, dim=1)

        # [N; comp; C; H; W]
        batch_r = batch.unsqueeze(1).repeat(1, self._n_comp, 1, 1, 1)

        # [N; C; H; W]
        dist = Normal(mu, log_s.exp())
        z = (w * dist.cdf(batch_r)).sum(dim=1).clip(0, 1)
        log_det = (w * dist.log_prob(batch_r).exp()).sum(dim=1).log() - self.__ln_2

        return z, log_det

    def inverse_flow(self, w: torch.Tensor, mu: torch.Tensor, log_s: torch.Tensor) -> torch.Tensor:
        bs, c, h, w = w.shape
        log_s = log_s.exp()

        z = self._base_dist.sample((bs,))

        output_pixels = torch.empty_like(w)
        for i in range(bs):
            dist = Normal(mu[i], log_s[i])

            def f(x: int):
                x = w.new_tensor([x] * self._n_comp)
                return w[i].dot(dist.cdf(x)) - z[i]

            output_pixels[i] = bisect(f, -20, 20)

        return output_pixels

    def sample(self, n_samples: int = 100) -> torch.Tensor:
        c, h, w = self.__pixel_cnn.input_shape
        with torch.no_grad():
            samples = torch.zeros((n_samples, c, h, w), device=self.__pixel_cnn.device)
            for y in range(h):
                for x in range(w):
                    for c in range(c):
                        pixel_out = self(samples)
                        w, mu, log_s = pixel_out.chunk(3, dim=1)
                        w = torch.softmax(w, dim=1)
                        samples[:, c, y, x] = self.pixel_inverse_flow(w, mu, log_s)

        return torch.clip(samples, 0, 1).permute(0, 2, 3, 1)
