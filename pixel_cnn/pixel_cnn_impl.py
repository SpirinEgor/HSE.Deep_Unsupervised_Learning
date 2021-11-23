from typing import Tuple

from torch import nn, Tensor

from pixel_cnn.masked_conv import ConvTypeB, ConvTypeA


class ResidualBlock(nn.Module):
    def __init__(self, n_channels: int):
        super().__init__()
        self.__model = nn.Sequential(
            ConvTypeB(in_channels=n_channels, out_channels=n_channels // 2, kernel_size=1),
            nn.ReLU(),
            ConvTypeB(in_channels=n_channels // 2, out_channels=n_channels // 2, kernel_size=7, padding=3),
            nn.ReLU(),
            ConvTypeB(in_channels=n_channels // 2, out_channels=n_channels, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.__model(x) + x


class PixelCNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        height: int,
        width: int,
        n_layers: int,
        n_filters: int,
    ):
        super().__init__()
        self._c, self._h, self._w = in_channels, height, width
        self._out = out_channels

        layers = [ConvTypeA(in_channels, n_filters, kernel_size=7, padding=3)]
        for _ in range(n_layers):
            layers.append(ResidualBlock(n_filters))
        layers.extend(
            [
                nn.ReLU(),
                ConvTypeB(in_channels=n_filters, out_channels=n_filters, kernel_size=1),
                nn.ReLU(),
                ConvTypeB(in_channels=n_filters, out_channels=in_channels * self._out, kernel_size=1),
            ]
        )

        self.__model = nn.Sequential(*layers)

    def forward(self, batch: Tensor) -> Tensor:
        """Forward pass of PixelCNN

        :param batch: tensor of shape [N; C; H; W]
        :return: tensor of shape [N; OUT; C; H; W ]
        """
        # [N; C * OUT; H; W]
        output = self.__model(batch)
        return output.reshape(batch.shape[0], self._out, self._c, self._h, self._w)

    @property
    def input_shape(self) -> Tuple[int, int, int]:
        return self._c, self._h, self._w
