from typing import Optional, Tuple, List

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm.auto import trange

from flow.flow_pixel_cnn import FlowPixelCNN


class Trainer:
    def __init__(
        self,
        mixture_components: int,
        img_channels: int,
        img_height: int,
        img_width: int,
        pixel_n_layers: int,
        pixel_n_filters: int,
    ):
        pixel_cnn_params = {
            "in_channels": img_channels,
            "height": img_height,
            "width": img_width,
            "n_layers": pixel_n_layers,
            "n_filters": pixel_n_filters,
        }

        self._device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print(f"Use {self._device} device")

        dist_params = [torch.zeros((1,), device=self._device), torch.ones((1,), device=self._device)]
        self._pixel_flow = FlowPixelCNN(mixture_components, pixel_cnn_params, dist_params).to(self._device)

    def _step(self, batch: torch.Tensor, is_train: bool = False) -> torch.Tensor:
        batch = batch.to(self._device)
        if is_train:
            noise = torch.rand(batch.shape, device=self._device)
            batch = (batch + noise) / 2

        # [N; C; H; W]
        z, log_det = self._pixel_flow(batch)
        # [N; C]
        log_prob = (self._pixel_flow.base_dist.log_prob(z) + log_det).mean(dim=(2, 3))

        return -log_prob.mean()

    def fit(
        self,
        train_dataloader: DataLoader,
        n_epochs: int,
        lr: float = 1e-3,
        clip_norm: Optional[float] = None,
        test_dataloader: Optional[DataLoader] = None,
    ) -> Tuple[List[float], ...]:
        optimizer = torch.optim.AdamW(self._pixel_flow.parameters(), lr=lr)

        train_losses, test_losses = [], []
        epoch_range = trange(n_epochs, desc="Epochs")
        for _ in epoch_range:
            self._pixel_flow.train()
            epoch_train_losses = []
            for batch in train_dataloader:
                loss = self._step(batch, is_train=True)
                optimizer.zero_grad()
                loss.backward()
                if clip_norm is not None:
                    clip_grad_norm_(self._pixel_flow.parameters(), 1)
                optimizer.step()

                epoch_train_losses.append(loss.item())
            epoch_loss = sum(epoch_train_losses) / len(epoch_train_losses)
            train_losses.extend(epoch_train_losses)

            if test_dataloader is not None:
                self._pixel_flow.eval()
                epoch_test_losses = []
                for batch in test_dataloader:
                    loss = self._step(batch, is_train=False)
                    epoch_test_losses.append(loss.item())
                test_loss = sum(epoch_test_losses) / len(epoch_test_losses)
                test_losses.append(test_loss)
                epoch_range.set_postfix({"train loss": epoch_loss, "test loss": test_loss})
            else:
                epoch_range.set_postfix({"train loss": epoch_loss})
        epoch_range.close()

        return (train_losses, test_losses) if test_dataloader is not None else train_losses

    @property
    def model(self) -> FlowPixelCNN:
        return self._pixel_flow

    @property
    def device(self) -> torch.device:
        return self._device
