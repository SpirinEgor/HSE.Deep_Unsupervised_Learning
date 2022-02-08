from typing import Tuple

import torch
from torch.utils.data import Dataset


class ContextDataset(Dataset):
    _img_h, _img_w = 28, 28
    _crop_h, _crop_w = 14, 14

    def __init__(self, data):
        self._data = data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # [1; h; w]
        image, _ = self._data[idx]
        # [1; h; w]
        mask = self.get_mask()
        target = image[mask].reshape(-1, self._crop_h, self._crop_w)
        image[mask] = 0
        return image, target

    @classmethod
    def get_mask(cls):
        mask = torch.zeros((1, cls._img_h, cls._img_w), dtype=torch.bool)
        h_c, w_c = cls._img_h // 2, cls._img_w // 2
        h_p, w_p = cls._crop_h // 2, cls._crop_w // 2
        mask[:, h_c - h_p : h_c + h_p, w_c - w_p : w_c + w_p] = True
        return mask
