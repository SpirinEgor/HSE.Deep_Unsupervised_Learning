import contextlib

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm.auto import trange, tqdm

from contrastive.modules import ImageEncoder


class ImageClassifier(nn.Module):
    def __init__(self, n_classes: int, latent_dim: int = 128, lr_cf: float = 1e-2):
        super().__init__()
        self.encoder = ImageEncoder(3, latent_dim)
        self.batch_norm = nn.BatchNorm1d(latent_dim)
        self.leaky_relu = nn.LeakyReLU(lr_cf)
        self.classifier = nn.Linear(latent_dim, n_classes)

    def forward(self, batched_images: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(batched_images)
        hidden = self.leaky_relu(self.batch_norm(encoded))
        return self.classifier(hidden)


class SemiSupervisedTrainer:
    def __init__(self, n_classes: int, device: torch.device, latent_dim: int = 128, u_loss_weight: float = 1.0):
        self.model = ImageClassifier(n_classes, latent_dim).to(device)
        self.device = device
        self.u_loss_weight = u_loss_weight

    def test(self, dataloader: DataLoader):
        self.model.eval()
        n_correct = 0
        total = 0
        for img, y in dataloader:
            img = img.to(self.device)
            y = y.to(self.device)
            with torch.no_grad():
                y_pred = self.model(img).argmax(dim=-1)
                n_correct += (y == y_pred).sum().item()
                total += y.shape[0]
        return n_correct / total

    def unlabeled_loss(self, batched_images: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def fit(self, train_dataloader: DataLoader, test_dataloader: DataLoader, n_epochs: int, lr: float = 1e-3):
        optim = AdamW(self.model.parameters(), lr=lr)

        train_losses = []
        test_scores = [self.test(test_dataloader)]
        tqdm.write(f"Start test score: {test_scores[-1]}")

        for i in trange(n_epochs, desc="Epochs"):
            train_tqdm = tqdm(train_dataloader, desc="Training", leave=False)
            postfix = {}
            self.model.train()
            for img, y in train_tqdm:
                img = img.to(self.device)
                y = y.to(self.device)

                unsupervised_loss = self.unlabeled_loss(img)

                labeled = y != -1
                model_output = self.model(img[labeled])
                ce_loss = F.cross_entropy(model_output, y[labeled])

                loss = ce_loss + self.u_loss_weight * unsupervised_loss

                optim.zero_grad()
                loss.backward()
                optim.step()

                train_losses.append(loss.item())
                postfix["loss"] = train_losses[-1]
                train_tqdm.set_postfix(postfix)
            train_tqdm.close()

            test_scores.append(self.test(test_dataloader))
            tqdm.write(f"Test score after epoch {i + 1}: {test_scores[-1]}")
        return train_losses, test_scores


class VAT(SemiSupervisedTrainer):
    def __init__(self, n_classes: int, device: torch.device, latent_dim: int = 128, xi: float = 10, alpha: float = 1.0):
        super().__init__(n_classes, device, latent_dim, alpha)
        self.xi = xi

    @staticmethod
    @contextlib.contextmanager
    def _disable_tracking_bn_stats(model):
        def switch_attr(m):
            if hasattr(m, "track_running_stats"):
                m.track_running_stats ^= True

        model.apply(switch_attr)
        yield
        model.apply(switch_attr)

    def unlabeled_loss(self, batched_images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            pred = F.softmax(self.model(batched_images), dim=1)

        # prepare random unit tensor
        d = torch.normal(mean=0, std=1, size=batched_images.shape, device=self.device, requires_grad=True)

        with self._disable_tracking_bn_stats(self.model):
            pred_hat = self.model(batched_images + self.xi * d)
            logp_hat = F.log_softmax(pred_hat, dim=1)
            adv_distance = F.kl_div(logp_hat, pred, reduction="batchmean")

            adv_distance.backward()
            r_adv = F.normalize(d.grad)
            self.model.zero_grad()

            pred_hat = self.model(batched_images + r_adv)
            logp_hat = F.log_softmax(pred_hat, dim=1)
            lds = F.kl_div(logp_hat, pred, reduction="batchmean")

        return lds


class FixMatch(SemiSupervisedTrainer):
    def __init__(
        self, n_classes: int, device: torch.device, latent_dim: int = 128, tau: float = 0.95, lambda_u: float = 1
    ):
        super().__init__(n_classes, device, latent_dim, lambda_u)
        self.tau = tau

        self.strong_transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(size=32),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)], p=0.8
                ),
                transforms.RandomGrayscale(0.2),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        self.weak_transforms = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()])

    def unlabeled_loss(self, batched_images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.model(self.weak_transforms(batched_images))
            probs = F.softmax(logits, dim=-1)
            confidence, pseudo_target = torch.max(probs, dim=-1)
            mask = confidence >= self.tau

        if mask.sum() == 0:
            return 0

        strong_out = self.model(self.strong_transforms(batched_images))
        loss = F.cross_entropy(strong_out[mask], pseudo_target[mask])
        return loss
