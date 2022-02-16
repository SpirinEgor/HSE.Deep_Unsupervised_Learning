from copy import deepcopy
from itertools import chain

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm.auto import trange, tqdm

from contrastive.modules import ImageEncoder, Predictor


class BYOL:
    def __init__(self, latent_dim: int, device: torch.device, m: float = 4e-3):
        self.student = ImageEncoder(latent_dim).to(device)

        self.teacher = deepcopy(self.student)
        for param in self.teacher.parameters():
            param.requires_grad = False

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

    @staticmethod
    def regression_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        norm_x = torch.linalg.norm(x, dim=-1)
        norm_y = torch.linalg.norm(y, dim=-1)
        return -2 * torch.mean(torch.sum(x * y, dim=-1) / (norm_x * norm_y))

    def update_teacher(self):
        for st_param, t_param in zip(self.student.parameters(), self.teacher.parameters()):
            t_param.data = t_param.data + (1 - self.m) * (st_param.data - t_param.data)

    def fit(self, train_dataloader: DataLoader, lr: float = 1e-3, n_epochs: int = 5):
        optim = torch.optim.AdamW(chain(self.student.parameters(), self.predictor.parameters()), lr=lr)
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

                student_out_1 = self.predictor(self.student(x))
                student_out_2 = self.predictor(self.student(y))
                with torch.no_grad():
                    teacher_out_1 = self.teacher(x)
                    teacher_out_2 = self.teacher(y)

                loss = self.regression_loss(student_out_1, teacher_out_2)
                loss += self.regression_loss(student_out_2, teacher_out_1)

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
