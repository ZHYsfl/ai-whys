import torch
from torch import nn
from torch.nn import functional as F


class WaveDiscriminator(nn.Module):
    '''
    单尺度判别器网络
    '''
    def __init__(self, base_channels: int = 16) -> None:
        super().__init__()
        if base_channels < 4 or base_channels % 4:
            raise ValueError("base_channels must be a multiple of 4")
        self.input_conv = nn.Conv1d(1, base_channels, kernel_size=15, padding=7) # 长度不变：(T + 2×7 − 15)/1 + 1 = T
        channels = [base_channels * 2, base_channels * 4, base_channels * 8]
        self.convs = nn.ModuleList()
        in_channels = base_channels
        for out_channels in channels:
            self.convs.append(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=41,
                    stride=4,
                    padding=20,
                    groups=4,
                ) # T' = (T + 2*20 − 41)/4 + 1 = T/4  
            )
            in_channels = out_channels
        self.output_conv = nn.Conv1d(in_channels, 1, kernel_size=5, padding=2) # 长度不变
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        # x.shape is (batch_size,1,time_steps) 也就是 [B, 1, T]
        features = []
        x = self.activation(self.input_conv(x))
        features.append(x)
        for conv in self.convs:
            x = self.activation(conv(x))
            features.append(x)
        score = self.output_conv(x) # (batch_size,1,new_time_steps=time_steps/4/4/4)
        return score, features


class MultiScaleDiscriminator(nn.Module):
    '''
    多尺度判别器网络
    '''
    def __init__(self, scales: tuple[int, ...] = (1, 2, 4), base_channels: int = 16) -> None:
        super().__init__()
        if any(scale < 1 for scale in scales):
            raise ValueError("scales must be positive")
        self.scales = scales
        self.discriminators = nn.ModuleList(
            [WaveDiscriminator(base_channels) for _ in scales]
        )

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
        all_scores = []
        all_features = []
        for scale, discriminator in zip(self.scales, self.discriminators):
            scaled = x if scale == 1 else F.avg_pool1d(x, kernel_size=scale, stride=scale)
            scores, features = discriminator(scaled)
            all_scores.append(scores)
            all_features.append(features)
        return all_scores, all_features


def discriminator_loss(real_scores: list[torch.Tensor], fake_scores: list[torch.Tensor]) -> torch.Tensor:
    if len(real_scores) != len(fake_scores) or not real_scores:
        raise ValueError("expected equally many, non-empty score lists")
    losses = []
    for real, fake in zip(real_scores, fake_scores):
        real_loss = F.relu(1 - real).mean()
        fake_loss = F.relu(1 + fake).mean()
        losses.append(real_loss + fake_loss)
    return torch.stack(losses).mean()


def generator_adversarial_loss(fake_scores: list[torch.Tensor]) -> torch.Tensor:
    if not fake_scores:
        raise ValueError("fake_scores cannot be empty")
    return torch.stack([F.relu(1 - fake).mean() for fake in fake_scores]).mean()


def feature_matching_loss(
    real_features: list[list[torch.Tensor]],
    fake_features: list[list[torch.Tensor]],
) -> torch.Tensor:
    if len(real_features) != len(fake_features) or not real_features:
        raise ValueError("expected equally many, non-empty feature lists")
    losses = []
    for real_scale, fake_scale in zip(real_features, fake_features):
        if len(real_scale) != len(fake_scale):
            raise ValueError("feature counts differ between real and fake inputs")
        for real, fake in zip(real_scale, fake_scale):
            losses.append(F.l1_loss(fake, real.detach()))
    return torch.stack(losses).mean()
