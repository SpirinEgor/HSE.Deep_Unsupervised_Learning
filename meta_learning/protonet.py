from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import trange, tqdm

from utils.hw14_utils import split_batch


class DownSampleBlock(nn.Module):
    def __init__(self, n_channels: int):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_channels, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.main(x)


class ProtoNet(nn.Module):
    def __init__(self, embedding_dim: int = 128, hidden_dim: int = 128):
        super().__init__()
        layers = [
            nn.Conv2d(1, hidden_dim, kernel_size=3, stride=1, padding=1),  # [1; 28; 28] -> [h; 28; 28]
            nn.BatchNorm2d(hidden_dim, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
        ]

        for _ in range(3):
            # [h; 28; 28] -> [h; 14; 14] -> [h; 7; 7] -> [h; 3; 3]
            layers.append(DownSampleBlock(hidden_dim))

        layers += [nn.Flatten(), nn.Linear(hidden_dim * 3 * 3, embedding_dim)]
        self.encoder = nn.Sequential(*layers)
        self.embedding_dim = embedding_dim

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def get_prototypes(self, data: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor]:
        unique_labels = torch.unique(labels)
        embeddings = self.encoder(data)

        mean = torch.zeros(len(unique_labels), self.embedding_dim, device=data.device)

        for i, label in enumerate(unique_labels):
            mean[i] = torch.mean(embeddings[labels == label], dim=0)

        return mean, unique_labels

    @torch.no_grad()
    def adapt_few_shots(self, batch, dloader):
        """
        :param batch: n-way_test k-shot_test batch (pair) of images ([k_shot_test * n-way_test, 1, 28, 28])
                  and labeles [k_shot_test * n-way_test])
        :param dloader: dataloader for the test set. yields batches of images ([batch_size, 1, 28, 28])
                    with their labelel ([batch_size])

        :returns pred: np.array of predicted classes for each images in dloader (don't shuffle it)
        """
        batch_imgs, batch_labels = [b.to(self.device) for b in batch]
        prototypes, unique_labels = self.get_prototypes(batch_imgs, batch_labels)

        pred = []
        for batch in tqdm(dloader):
            imgs, labels = [b.to(self.device) for b in batch]

            pred_idxs = torch.argmin(pairwise_dist(self.encoder(imgs), prototypes), dim=-1)
            pred.append(unique_labels[pred_idxs])

        return torch.stack(pred).detach().cpu().numpy()


def pairwise_dist(x: Tensor, y: Tensor) -> Tensor:
    return torch.sum(x ** 2, dim=1, keepdim=True) + torch.sum(y ** 2, dim=1) - 2 * x @ y.T


def train(protonet: ProtoNet, train_dataloader: DataLoader, device: torch.device, n_epochs: int = 5, lr: float = 1e-3):
    optim = AdamW(protonet.parameters(), lr=lr)

    losses = []
    for _ in trange(n_epochs, desc="Epochs"):
        batches_pbar = tqdm(train_dataloader, desc="Training", leave=False)
        for batch in batches_pbar:
            batch = [it.to(device) for it in batch]

            train_imgs, test_imgs, train_labels, test_labels = split_batch(*batch)

            prototypes, unique_labels = protonet.get_prototypes(train_imgs, train_labels)
            _, test_labels_idx = torch.unique(test_labels, return_inverse=True)

            dist = pairwise_dist(protonet(test_imgs), prototypes)
            log_p = F.log_softmax(-dist, dim=-1)
            log_p = log_p[torch.arange(len(log_p)), test_labels_idx]

            loss = torch.tensor([0.0], device=device)
            for label in unique_labels:
                label_idx = test_labels == label
                # J + J_class / (N_q * N_c)
                loss = loss + log_p[label_idx].sum() / (label_idx.sum() * len(unique_labels))
            loss = -loss

            optim.zero_grad()
            loss.backward()
            optim.step()

            losses.append(loss.item())
            batches_pbar.set_postfix({"loss": losses[-1]})

        batches_pbar.close()
    return losses
