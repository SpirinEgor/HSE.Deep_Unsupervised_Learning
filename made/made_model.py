from typing import List, Tuple

import torch
import numpy as np
from torch import log_softmax, softmax
from torch.nn import ReLU, Sequential, Module

from made.masked_linear import MaskedLinear


class MADE(Module):
    """Implements Masked AutoEncoder for Density Estimation, by Germain et al. 2015
    Paper link: https://arxiv.org/abs/1502.03509
    """

    def __init__(
        self,
        n_features: int,
        d_size: int,
        hidden_sizes: List[int] = None,
        use_one_hot: bool = False,
    ):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = []
        self.__use_one_hot = use_one_hot
        self.__n_features = n_features
        self.__input_dim = n_features * d_size if use_one_hot else n_features
        self.__layer_sizes = [self.__input_dim] + hidden_sizes + [n_features * d_size]
        self.__d_size = d_size

        self.__layers = []
        for h_in, h_out in zip(self.__layer_sizes, self.__layer_sizes[1:]):
            self.__layers += [MaskedLinear(h_in, h_out), ReLU()]
        self.__layers.pop()
        self.__made = Sequential(*self.__layers)

        self.__init_masks()

    def __init_masks(self):
        n_layers = len(self.__layer_sizes) - 2
        prev_order = np.arange(self.__n_features)
        masks = []
        for _l in range(n_layers):
            cur_order = np.random.randint(prev_order.min(), self.__input_dim - 1, size=self.__layer_sizes[_l + 1])
            mask = prev_order[:, None] <= cur_order[None, :]

            if _l == 0 and self.__use_one_hot:
                mask = np.repeat(mask, self.__d_size, axis=0)

            masks.append(mask)
            prev_order = cur_order

        last_layer_mask = prev_order[:, None] < np.arange(self.__n_features)[None, :]
        last_layer_mask = np.repeat(last_layer_mask, self.__d_size, axis=1)
        masks.append(last_layer_mask)

        layers = [_l for _l in self.__layers if isinstance(_l, MaskedLinear)]
        assert len(layers) == len(masks)

        for _m, _l in zip(masks, layers):
            _l.set_mask(_m)

    def forward(self, input_batch: torch.Tensor) -> torch.Tensor:
        """MADE forward pass.
        :param input_batch: [batch size; *n features] tensor with input features
        :return [batch size; d size; *n features] tensor with features distributions
        """
        batch_size = input_batch.shape[0]

        if self.__use_one_hot:
            input_batch_flat = input_batch.view(-1)
            ohe_features = input_batch.new_zeros((input_batch_flat.shape[0], self.__d_size))
            ohe_features[torch.arange(input_batch_flat.shape[0]), input_batch_flat] = 1
            # [batch size; d * input shape size] = [batch size; k]
            made_input = ohe_features.view(batch_size, -1).float()
        else:
            # [batch size; input shape size] = [batch size; k]
            made_input = input_batch.view(batch_size, -1).float()

        logits = self.__made(made_input).view(batch_size, -1, self.__d_size)
        return logits.permute(0, 2, 1).view(batch_size, self.__d_size, *input_batch.shape[1:])

    def get_distribution(self) -> torch.Tensor:
        if self.__n_features != 2:
            raise RuntimeError("Distribution building only supported for 2D joint")
        x = np.mgrid[0 : self.__d_size, 0 : self.__d_size].reshape(2, self.__d_size ** 2).T
        x = torch.tensor(np.ascontiguousarray(x), dtype=torch.long, device=self.device, requires_grad=False)
        log_probabilities = log_softmax(self(x), dim=1)
        distribution = torch.gather(log_probabilities, 1, x.unsqueeze(1)).squeeze(1)
        distribution = distribution.sum(dim=1)
        return distribution.exp().view(self.__d_size, self.__d_size).detach().cpu().numpy()

    def sample(self, n: int, result_shape: Tuple) -> np.ndarray:
        if np.prod(result_shape) != self.__n_features:
            raise ValueError(f"Result shape ({result_shape}) mismatch size of input features ({self.__n_features})")
        samples = torch.zeros(n, self.__n_features, device=self.device)
        inv_ordering = {x: i for i, x in enumerate(np.arange(self.__input_dim))}
        with torch.no_grad():
            for i in range(self.__n_features):
                logits = self(samples).view(n, self.__d_size, self.__n_features)[:, :, inv_ordering[i]]
                probabilities = softmax(logits, dim=1)
                samples[:, inv_ordering[i]] = torch.multinomial(probabilities, 1).squeeze(-1)
        return samples.view(n, *result_shape).cpu().numpy()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
