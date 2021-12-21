import torch
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader
from tqdm import trange

from pixel_cnn import PixelCNN


def train_pixel_cnn(
    pixel_cnn: PixelCNN,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    device: torch.device,
    epochs: int = 20,
    lr: float = 1e-3,
):
    pixel_cnn = pixel_cnn.to(device)
    optim = torch.optim.AdamW(pixel_cnn.parameters(), lr=lr)

    train_losses = []
    test_losses = [test_pixel_cnn(pixel_cnn, test_dataloader, device)]

    epoch_bar = trange(epochs, desc="PixelCNN training")
    postfix = {"test_loss": test_losses[-1]}
    for _ in epoch_bar:
        for batch in train_dataloader:
            batch = batch.to(device).squeeze(1).long()
            out = pixel_cnn(batch)
            loss = cross_entropy(out, batch.detach())

            optim.zero_grad()
            loss.backward()
            optim.step()

            train_losses.append(loss.item())
            postfix["train_loss"] = train_losses[-1]
            epoch_bar.set_postfix(postfix)

        test_losses.append(test_pixel_cnn(pixel_cnn, test_dataloader, device))
        postfix["test_loss"] = test_losses[-1]
        epoch_bar.set_postfix(postfix)
    epoch_bar.close()

    return train_losses, test_losses


def test_pixel_cnn(pixel_cnn: PixelCNN, test_dataloader: DataLoader, device: torch.device) -> float:
    pixel_cnn.eval()
    loss = 0.0
    for batch in test_dataloader:
        batch = batch.to(device).squeeze(1).long()
        out = pixel_cnn(batch)
        cur_loss = cross_entropy(out, batch.detach())
        loss += cur_loss.item()
    return loss / len(test_dataloader)
