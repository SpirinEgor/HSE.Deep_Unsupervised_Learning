from copy import deepcopy
from itertools import chain

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm.auto import trange, tqdm

from contrastive.modules import ImageEncoder, Predictor


class BYOL:
    def __init__(self, latent_dim: int, device: torch.device, m: float = 0.99):
        self.student = ImageEncoder(latent_dim).to(device)

        self.teacher = deepcopy(self.student)
        for st_param, t_param in zip(self.student.parameters(), self.teacher.parameters()):
            t_param.data.copy_(st_param.data)
            t_param.requires_grad = False

        self.predictor = Predictor(latent_dim).to(device)

        self.transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(size=28),
                transforms.GaussianBlur(5, sigma=(0.1, 2.0)),
            ]
        )

        self.device = device
        self.m = m

    def step(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self.predictor(self.student(x))
        norm_x = F.normalize(x, dim=1)

        with torch.no_grad():
            y = self.teacher(y)
            norm_y = F.normalize(y, dim=1)

        loss = 2 - 2 * (norm_x * norm_y).sum(dim=-1)
        return loss

    def update_teacher(self):
        for st_param, t_param in zip(self.student.parameters(), self.teacher.parameters()):
            t_param.data = t_param.data * self.m + st_param.data * (1 - self.m)

    def fit(self, train_dataloader: DataLoader, lr: float = 3e-4, n_epochs: int = 5):
        optim = torch.optim.AdamW(
            chain(self.student.parameters(), self.predictor.parameters()), lr=lr, weight_decay=1e-4
        )
        losses = []
        self.student.train()
        self.predictor.train()
        for e in trange(n_epochs, desc="Epochs"):
            is_last_epoch = e == n_epochs - 1
            train_tqdm_bar = tqdm(train_dataloader, desc="Training", leave=is_last_epoch)
            postfix = {}
            for batch in train_tqdm_bar:
                x = self.transforms(batch).to(self.device)
                y = self.transforms(batch).to(self.device)

                loss = self.step(x, y) + self.step(y, x)
                loss = loss.mean()

                optim.zero_grad()
                loss.backward()
                optim.step()

                self.update_teacher()

                losses.append(loss.item())
                postfix["loss"] = losses[-1]
                train_tqdm_bar.set_postfix(postfix)
            train_tqdm_bar.close()
        return losses

    def encode(self, x: torch.Tensor):
        self.student.eval()
        with torch.no_grad():
            x = x.to(self.device)
            return self.student(x).detach()
