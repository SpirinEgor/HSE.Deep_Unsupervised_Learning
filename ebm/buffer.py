from typing import Tuple

import torch
from numpy.random import choice


class Buffer:
    def __init__(self, max_size: int, device: torch.device, sizes: Tuple[int, ...] = (1, 28, 28)):
        self._max_size = max_size
        self._device = device

        self._buffer = torch.rand((max_size, *sizes), dtype=torch.float, device=device) * 2 - 1
        self._cur_size = 0
        self._insert_ptr = 0

    def push(self, data: torch.Tensor):
        data = data.to(self._device)
        tail_size = min(data.shape[0], self._max_size - self._insert_ptr)
        self._buffer[self._insert_ptr : self._insert_ptr + tail_size] += data[:tail_size]
        self._buffer[: data.shape[0] - tail_size] = data[tail_size:]
        self._insert_ptr = (self._insert_ptr + data.shape[0]) % self._max_size
        self._cur_size = min(self._cur_size + data.shape[0], self._max_size)

    def sample(self, size: int, use_only_inserted: bool = False) -> torch.Tensor:
        limit = self._cur_size if use_only_inserted else self._max_size
        indices = choice(limit, size, replace=False)
        return self._buffer[indices]
