# Copyright (c) Kyutai, all rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Derived from kyutai-labs/moshi and AudioCraft; see NOTICE and LICENSES/.

from math import prod

from torch import nn

from .convolution import StreamingConv1d, StreamingConvTranspose1d
from .streaming import StreamingContainer


class SEANetResnetBlock(StreamingContainer):
    def __init__(
        self,
        dim: int,
        kernel_sizes: tuple[int, ...] = (3, 1),
        dilations: tuple[int, ...] = (1, 1),
        compress: int = 2,
        norm: str = "none",
    ) -> None:
        super().__init__()
        if len(kernel_sizes) != len(dilations):
            raise ValueError("kernel_sizes and dilations must have equal lengths")
        hidden = dim // compress
        layers = []
        for index, (kernel_size, dilation) in enumerate(zip(kernel_sizes, dilations, strict=True)):
            in_channels = dim if index == 0 else hidden
            out_channels = dim if index == len(kernel_sizes) - 1 else hidden
            layers.extend(
                [
                    nn.ELU(),
                    StreamingConv1d(
                        in_channels,
                        out_channels,
                        kernel_size,
                        dilation=dilation,
                        norm=norm,
                    ),
                ] # (t+ k-s -k)/s + 1 = t/s = t 长度不变
            )
        self.block = nn.Sequential(*layers)
        self.shortcut = nn.Identity()

    def forward(self, x):
        return self.shortcut(x) + self.block(x)


class SEANetEncoder(StreamingContainer):
    def __init__(
        self,
        channels: int = 1,
        dimension: int = 64,
        n_filters: int = 16,
        n_residual_layers: int = 1,
        ratios: tuple[int, ...] = (8, 6, 5, 4),
        norm: str = "none",
        compress: int = 2,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.dimension = dimension
        self.ratios = tuple(reversed(ratios))
        self.hop_length = prod(self.ratios)
        layers = [StreamingConv1d(channels, n_filters, 7, norm=norm)]
        multiplier = 1
        for ratio in self.ratios:
            for residual_index in range(n_residual_layers):
                layers.append(
                    SEANetResnetBlock(
                        multiplier * n_filters,
                        dilations=(2**residual_index, 1),
                        compress=compress,
                        norm=norm,
                    )
                )
            layers.extend(
                [
                    nn.ELU(),
                    StreamingConv1d(
                        multiplier * n_filters,
                        multiplier * n_filters * 2,
                        2 * ratio,
                        stride=ratio,
                        norm=norm,
                    ), # (t+ k-s -k)/s + 1 = t/s 时间缩短 ratio 倍，通道翻倍
                ]
            )
            multiplier *= 2
        layers.extend(
            [nn.ELU(), StreamingConv1d(multiplier * n_filters, dimension, 3, norm=norm)] # （t+k-s-k)/s+1 = t/s = t 长度不变
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # x.shape is (batch, 1, time_steps)
        return self.model(x)


class SEANetDecoder(StreamingContainer):
    def __init__(
        self,
        channels: int = 1,
        dimension: int = 64,
        n_filters: int = 16,
        n_residual_layers: int = 1,
        ratios: tuple[int, ...] = (8, 6, 5, 4),
        norm: str = "none",
        compress: int = 2,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.dimension = dimension
        self.ratios = ratios
        self.hop_length = prod(ratios)
        multiplier = 2 ** len(ratios)
        layers = [StreamingConv1d(dimension, multiplier * n_filters, 7, norm=norm)]
        for ratio in ratios:
            layers.extend(
                [
                    nn.ELU(),
                    StreamingConvTranspose1d(
                        multiplier * n_filters,
                        multiplier * n_filters // 2,
                        2 * ratio,
                        stride=ratio,
                        norm=norm,
                    ),
                ]
            )
            for residual_index in range(n_residual_layers):
                layers.append(
                    SEANetResnetBlock(
                        multiplier * n_filters // 2,
                        dilations=(2**residual_index, 1),
                        compress=compress,
                        norm=norm,
                    )
                )
            multiplier //= 2
        layers.extend([nn.ELU(), StreamingConv1d(n_filters, channels, 3, norm=norm)])
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
