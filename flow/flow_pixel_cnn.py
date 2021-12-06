from typing import Dict, Any, Tuple, Optional, List

import torch
from scipy.optimize import bisect
from torch import nn
from torch.distributions import Uniform, Normal, Distribution
from tqdm.auto import trange

from pixel_cnn import PixelCNN


class FlowPixelCNN(nn.Module):
    def __init__(self, mixture_components: int, pixel_cnn_kwargs: Dict[str, Any], dist_params: List[torch.Tensor]):
        super().__init__()
        self._n_comp = mixture_components
        output = 3 * mixture_components
        self.__pixel_cnn = PixelCNN(out_channels=output, **pixel_cnn_kwargs)

        self._base_dist = Uniform(*dist_params)

        self.__ln_2 = nn.Parameter(torch.tensor(2).log(), requires_grad=False)

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
        bs = w.shape[0]

        z = self._base_dist.sample((bs,))

        output_pixels = w.new_empty((bs,))
        for i in range(bs):
            dist = Normal(mu[i], log_s[i].exp())

            def f(x: int):
                x = w.new_tensor([x] * self._n_comp)
                return w[i].dot(dist.cdf(x)) - z[i]

            output_pixels[i] = bisect(f, -20, 20)

        return output_pixels

    def sample(self, n_samples: int = 100, device: Optional[torch.device] = None) -> torch.Tensor:
        c, h, w = self.__pixel_cnn.input_shape
        with torch.no_grad():
            samples = torch.zeros((n_samples, c, h, w), device=device)
            sampling_progress_bar = trange(h * w * c, desc="Sampling")
            for y in range(h):
                for x in range(w):
                    for z in range(c):
                        pixel_out = self.__pixel_cnn(samples)[:, :, z, y, x]
                        _w, mu, log_s = pixel_out.chunk(3, dim=1)
                        _w = torch.softmax(_w, dim=1)
                        samples[:, z, y, x] = self.inverse_flow(_w, mu, log_s)
                        sampling_progress_bar.update()
            sampling_progress_bar.close()

        return torch.clip(samples, 0, 1).permute(0, 2, 3, 1)

    @property
    def base_dist(self) -> Distribution:
        return self._base_dist
