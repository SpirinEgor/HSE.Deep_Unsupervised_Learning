from typing import Tuple, List, Union, Optional

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import trange, tqdm

from made.made_model import MADE
from utils.pytorch_utils import fix_everything

TRAIN_RETURN = Union[List[float], Tuple[List[float], List[float]]]


class ImageMADETrainer:
    def __init__(
        self,
        input_shape: Tuple[int, ...],
        n_discrete: int,
        hidden_sizes: List[int] = None,
        use_one_hot: bool = False,
    ):
        self.__use_one_hot = use_one_hot
        self.__d = n_discrete

        self.__device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.__made = MADE(np.prod(input_shape), n_discrete, hidden_sizes, use_one_hot=use_one_hot).to(self.__device)
        self.__loss = CrossEntropyLoss().to(self.__device)

    def train(
        self,
        train_dataloader: DataLoader,
        n_epochs: int,
        lr: float = 0.001,
        weight_decay: float = 0.0001,
        clip_norm: Optional[float] = None,
        seed: Optional[int] = None,
        test_dataloader: Optional[DataLoader] = None,
    ) -> TRAIN_RETURN:
        fix_everything(seed)
        optimizer = AdamW(self.__made.parameters(), lr=lr, weight_decay=weight_decay)
        train_losses, test_losses = [], []
        for epoch in trange(n_epochs, desc="Epoch"):
            self.__made.train()
            train_iterator = tqdm(train_dataloader, desc="Train")
            for batch in train_iterator:
                batch = batch.to(device=self.__device)
                self.__made.zero_grad()

                logits = self.made(batch)
                loss = self.__loss(logits, batch)

                loss.backward()
                if clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.__made.parameters(), clip_norm)
                optimizer.step()

                train_losses.append(loss.item())
                train_iterator.set_postfix({"loss": train_losses[-1]})
            train_iterator.close()

            if test_dataloader is None:
                continue

            self.__made.eval()
            epoch_test_losses = []
            for batch in tqdm(test_dataloader, desc="Test"):
                batch = batch.to(device=self.__device)
                logits = self.made(batch)
                loss = self.__loss(logits, batch)
                epoch_test_losses.append(loss.item())
            mean_loss: float = np.mean(epoch_test_losses)
            tqdm.write(f"Epoch #{epoch}\n\tmean test loss: {mean_loss}")
            test_losses.append(mean_loss)

        if test_dataloader is None:
            return train_losses
        return train_losses, test_losses

    @property
    def made(self) -> MADE:
        return self.__made
