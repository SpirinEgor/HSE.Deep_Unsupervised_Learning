from typing import Tuple, List, Union

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import trange, tqdm

from made.made import MADE


TRAIN_RETURN = Union[List[float], Tuple[List[float], List[float]]]


class ImageMADETrainer:
    def __init__(
        self,
        input_shape: Tuple[int, ...],
        hidden_sizes: List[int] = None,
        use_one_hot: bool = False,
        n_discrete: int = None,
    ):
        if use_one_hot and n_discrete is None:
            raise ValueError("n_discrete must be number in case of one-hot encoding.")
        self.__use_one_hot = use_one_hot
        self.__d = n_discrete

        n_features = np.prod(input_shape)[0]
        if use_one_hot:
            n_features *= n_discrete

        self.__device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.__made = MADE(n_features, n_discrete, hidden_sizes).to(self.__device)
        self.__loss = CrossEntropyLoss().to(self.__device)

    def forward_step(self, batch: np.ndarray) -> torch.Tensor:
        """One step of MADE

        :param batch: [batch size; input shape size]
        :return [batch size; input shape size; d size] logits with distribution of each input feature
        """
        batch_size = batch.shape[0]

        if self.__use_one_hot:
            ohe_features = np.zeros((batch.size, self.__d))
            ohe_features[np.arange(batch.size), batch] = 1
            # [batch size; d * input shape size] = [batch size; k]
            made_input = torch.tensor(ohe_features.view(batch_size, -1), device=self.__device)
        else:
            # [batch size; input shape size] = [batch size; k]
            made_input = torch.tensor(batch.view(batch_size, -1), device=self.__device)

        # [batch size; k; d]
        logits = self.__made(made_input)
        return logits

    def train(
        self,
        train_dataloader: DataLoader,
        n_epochs: int,
        lr: float = 0.001,
        weight_decay: float = 0.0001,
        clip_norm: float = None,
        test_dataloader: DataLoader = None,
    ) -> TRAIN_RETURN:
        optimizer = AdamW(self.__made.parameters(), lr=lr, weight_decay=weight_decay)
        train_losses, test_losses = [], []
        for epoch in trange(n_epochs, "Epoch"):
            self.__made.train()
            train_iterator = tqdm(train_dataloader, "Train", leave=False)
            for batch in train_iterator:
                batch = batch.to(device=self.__device)
                self.__made.zero_grad()

                logits = self.forward_step(batch)
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
            for batch in tqdm(test_dataloader, "Test", leave=False):
                batch = batch.to(device=self.__device)
                logits = self.forward_step(batch)
                loss = self.__loss(logits, batch)
                epoch_test_losses.append(loss.item())
            mean_loss: float = np.mean(epoch_test_losses)
            tqdm.write(f"Epoch #{epoch}: mean test loss: {mean_loss}")
            test_losses.append(mean_loss)

        if test_dataloader is None:
            return train_losses
        return train_losses, test_losses
