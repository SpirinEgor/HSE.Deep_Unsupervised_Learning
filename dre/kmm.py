import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import trange


class KernelMeanMatching(nn.Module):
    def __init__(self, dim=1, sigma=1, hd=128):
        super().__init__()
        self._net = nn.Sequential(
            nn.Linear(dim, hd), nn.ReLU(), nn.Linear(hd, hd), nn.ReLU(), nn.Linear(hd, 1), nn.Softplus()
        )
        self._sigma = sigma

    def forward(self, input_batch):
        return self._net(input_batch)

    def get_kernel(self, batch_x, batch_y):
        to_exp = (batch_x ** 2).sum(dim=1, keepdim=True) + (batch_y ** 2).sum(dim=1) - 2 * batch_x @ batch_y.T
        to_exp /= -2 * (self._sigma ** 2)
        return to_exp.exp()


class KMMTrainer:
    def __init__(self, kmm: KernelMeanMatching, device: torch.device):
        self._kmm = kmm
        self._device = device

    def fit(self, dataloader_num: DataLoader, dataloader_den: DataLoader, lr: float = 1e-3, epochs: int = 1000):
        optim = torch.optim.AdamW(self._kmm.parameters(), lr=lr)
        self._kmm.train()

        epochs_iter = trange(epochs)
        postfix = {}
        for _ in epochs_iter:
            epoch_loss, iters = 0, 0
            for (batch_num, batch_den) in zip(dataloader_num, dataloader_den):
                batch_num = batch_num.to(self._device)
                batch_den = batch_den.to(self._device)

                r = self._kmm(batch_den)
                kernel_den_den = self._kmm.get_kernel(batch_den, batch_den)
                kernel_den_num = self._kmm.get_kernel(batch_den, batch_num)

                loss = r.T.matmul(kernel_den_den).matmul(r) - 2 * r.T.matmul(kernel_den_num).sum()
                loss /= batch_num.shape[0] * batch_num.shape[0]

                optim.zero_grad()
                loss.backward()
                optim.step()

                epoch_loss += loss.item()
                iters += 1

            postfix["loss"] = epoch_loss / iters
            epochs_iter.set_postfix(postfix)

        epochs_iter.close()
