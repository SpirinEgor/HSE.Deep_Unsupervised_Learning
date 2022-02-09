from itertools import chain

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import trange

from context_encoder.modules import ImageEncoder


class RotationPredictModel(nn.Module):
    def __init__(self, latent_dim: int, n_rotation: int, device: torch.device):
        super().__init__()
        self.encoder = ImageEncoder(latent_dim).to(device)
        self.classifier = nn.Linear(latent_dim, n_rotation).to(device)
        self.device = device

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.encoder(images)
        return self.classifier(features)

    def fit(self, train_dataloader: DataLoader, n_epochs: int, lr: float):
        optim = torch.optim.AdamW(chain(self.encoder.parameters(), self.classifier.parameters()), lr=lr)

        losses = []
        accuracies = []

        epoch_bar = trange(n_epochs, desc="Epochs")
        for _ in range(n_epochs):
            self.encoder.train()
            self.classifier.train()
            for i, (images, targets) in enumerate(train_dataloader):
                images = images.to(self.device)
                targets = targets.to(self.device)

                classes_logits = self(images)
                loss = F.cross_entropy(classes_logits, targets)

                optim.zero_grad()
                loss.backward()
                optim.step()

                losses.append(loss.item())

                predicted_rot = classes_logits.argmax(dim=-1)
                acc = (predicted_rot == targets).sum() / targets.shape[0]
                accuracies.append(acc.item())

                postfix = {"batch": f"{i + 1}/{len(train_dataloader)}", "loss": losses[-1], "acc": accuracies[-1]}
                epoch_bar.set_postfix(postfix)
        epoch_bar.close()
        return losses, accuracies
