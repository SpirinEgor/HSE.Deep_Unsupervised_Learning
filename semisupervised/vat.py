import contextlib

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
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


class VAT:
    def __init__(
        self, n_classes: int, device: torch.device, vat_alpha: float = 1.0, vat_xi: float = 10, latent_dim: int = 128
    ):
        self.model = ImageClassifier(n_classes, latent_dim).to(device)
        self.device = device

        self.vat_alpha = vat_alpha
        self.vat_eps = vat_xi

    @staticmethod
    @contextlib.contextmanager
    def _disable_tracking_bn_stats(model):
        def switch_attr(m):
            if hasattr(m, "track_running_stats"):
                m.track_running_stats ^= True

        model.apply(switch_attr)
        yield
        model.apply(switch_attr)

    def val_loss(self, batched_images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            pred = F.log_softmax(self.model(batched_images), dim=1)

        # prepare random unit tensor
        d = torch.normal(mean=0, std=1, size=batched_images.shape, device=self.device)

        with self._disable_tracking_bn_stats(self.model):
            # calc adversarial direction
            d.requires_grad_()
            pred_hat = self.model(batched_images + self.vat_eps * d)
            logp_hat = F.log_softmax(pred_hat, dim=1)
            adv_distance = F.kl_div(logp_hat, pred, reduction="batchmean")
            adv_distance.backward()
            r_adv = d.grad
            self.model.zero_grad()

            # calc LDS
            pred_hat = self.model(batched_images + r_adv)
            logp_hat = F.log_softmax(pred_hat, dim=1)
            lds = F.kl_div(logp_hat, pred, reduction="batchmean")

        return lds

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

    def fit(self, train_dataloader: DataLoader, test_dataloader: DataLoader, n_epochs: int, lr: float = 1e-4):
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

                lds = self.val_loss(img)

                labeled = y != -1
                model_output = self.model(img[labeled])
                ce_loss = F.cross_entropy(model_output, y[labeled])

                loss = ce_loss + self.vat_alpha * lds

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
