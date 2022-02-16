import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm.auto import trange, tqdm

from contrastive.modules import ImageEncoder


class BarlowTwins:
    def __init__(self, latent_dim: int, device: torch.device, lmbda: float = 0.01):
        self.encoder = ImageEncoder(3, latent_dim).to(device)
        self.transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(size=28),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)], p=0.8
                ),
                transforms.GaussianBlur(kernel_size=9),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        self.device = device
        self.lmbda = lmbda

    def step(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # compute embeddings
        z_a = self.encoder(a)  # NxD
        z_b = self.encoder(b)  # NxD
        bs, dim = z_a.shape

        # normalize repr. along the batch dimension
        z_a_norm = (z_a - z_a.mean(0)) / z_a.std(0)  # NxD
        z_b_norm = (z_b - z_b.mean(0)) / z_b.std(0)  # NxD

        # cross-correlation matrix
        c = torch.mm(z_a_norm.T, z_b_norm) / bs  # DxD

        # loss
        c_diff = (c - torch.eye(dim)).pow(2)  # DxD

        # multiply off-diagonal elems of c_diff by lambda
        c_diff *= self.lmbda
        torch.diagonal(c_diff).mul_(1.0 / self.lmbda)

        loss = c_diff.sum()
        return loss

    def fit(self, train_dataloader: DataLoader, lr: float = 3e-4, n_epochs: int = 5):
        optim = torch.optim.AdamW(self.encoder.parameters(), lr=lr, weight_decay=1e-4)
        losses = []
        self.encoder.train()
        for e in trange(n_epochs, desc="Epochs"):
            is_last_epoch = e == n_epochs - 1
            train_tqdm_bar = tqdm(train_dataloader, desc="Training", leave=is_last_epoch)
            postfix = {}
            for batch in train_tqdm_bar:
                x = self.transforms(batch).to(self.device)
                y = self.transforms(batch).to(self.device)

                loss = self.step(x, y)
                optim.zero_grad()
                loss.backward()
                optim.step()

                losses.append(loss.item())
                postfix["loss"] = losses[-1]
                train_tqdm_bar.set_postfix(postfix)
            train_tqdm_bar.close()
        return losses

    def encode(self, x: torch.Tensor):
        self.encoder.eval()
        with torch.no_grad():
            x = x.to(self.device)
            return self.encoder(x).detach()
