import math

import torch
from torch import nn
from torch.nn import functional as F

from .config import ASRConfig
from .features import lengths_to_mask


class Conv2dSubsampling(nn.Module):
    def __init__(self, config: ASRConfig) -> None:
        super().__init__()
        channels = config.d_model # default 64
        self.conv = nn.Sequential( # (batch_size, 1, num_window, n_mels)
            nn.Conv2d(1, channels, kernel_size=3, stride=2),
            # (n+2*0-3)//2+1 = (n-1)//2 
            # (batch_size, channels, (num_window-1)//2, (n_mels-1)//2)
            nn.SiLU(), # silu(x) = x * sigmoid(x) = x / (1 + exp(-x))

            nn.Conv2d(channels, channels, kernel_size=3, stride=2),
            # (n+2*0-3)//2+1 = (n-1)//2
            # (batch_size, channels, ((num_window-1)//2-1)//2, ((n_mels-1)//2-1)//2)
            nn.SiLU(),
        )
        self.projection = nn.Linear(channels * config.subsampled_mels, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        features: torch.Tensor, # (batch_size, num_window, n_mels)
        frame_lengths: torch.Tensor # (batch_size,) 每条音频的有效帧数
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output_lengths = frame_lengths
        for _ in range(2):
            output_lengths = (output_lengths - 1) // 2
        if (output_lengths < 1).any():
            raise ValueError("an utterance is too short for 4x convolutional subsampling")

        x = self.conv(features[:, None, :, :]) # (batch_size, channels, time_steps = ((num_window-1)//2-1)//2, frequency_steps = ((n_mels-1)//2-1)//2)
        batch_size, channels, time_steps, frequency_steps = x.shape
        x = x.transpose(1, 2).contiguous().view(
            batch_size, time_steps, channels * frequency_steps
        ) # x.shape is (batch_size, time_steps, channels * frequency_steps)
        x = self.dropout(self.projection(x)) # x.shape is (batch_size, time_steps, d_model)
        mask = lengths_to_mask(output_lengths, time_steps) #  mask.shape is (batch_size, time_steps)
        return x.masked_fill(~mask[:, :, None], 0.0), output_lengths, mask


class RoPEMultiHeadAttention(nn.Module):
    def __init__(self, config: ASRConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads # default 4
        self.head_dim = config.d_model // config.num_heads # default 64 // 4 = 16
        self.dropout = config.dropout # default 0.1
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.output = nn.Linear(config.d_model, config.d_model, bias=False)
        inv_freq = 1.0 / (
            10_000.0 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        ) # (head_dim // 2,)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def apply_rope(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape is (batch_size, num_heads, time_steps, head_dim)
        time_steps = x.shape[-2]
        positions = torch.arange(time_steps, device=x.device, dtype=self.inv_freq.dtype)
        angles = torch.outer(positions, self.inv_freq).to(x.dtype) # (time_steps, head_dim // 2)
        cos, sin = angles.cos()[None, None], angles.sin()[None, None]
        # cos.shape is (1, 1, time_steps, head_dim // 2)
        # sin.shape is (1, 1, time_steps, head_dim // 2)
        even, odd = x[..., 0::2], x[..., 1::2]
        # even.shape is (batch_size, num_heads, time_steps, head_dim // 2)
        # odd.shape is (batch_size, num_heads, time_steps, head_dim // 2)
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2) # (batch_size, num_heads, time_steps, head_dim // 2) -> (batch_size, num_heads, time_steps, head_dim//2, 2) -> (batch_size, num_heads, time_steps, head_dim)


    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x.shape is (batch_size, time_steps, d_model)
        # mask.shape is (batch_size, time_steps)
        batch_size, time_steps, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1) # q.shape, k.shape, v.shape is (batch_size, time_steps, d_model)
        q = q.view(batch_size, time_steps, self.num_heads, self.head_dim).transpose(1, 2) # (batch_size, num_heads, time_steps, head_dim)
        k = k.view(batch_size, time_steps, self.num_heads, self.head_dim).transpose(1, 2) # (batch_size, num_heads, time_steps, head_dim)
        v = v.view(batch_size, time_steps, self.num_heads, self.head_dim).transpose(1, 2) # (batch_size, num_heads, time_steps, head_dim)
        q, k = self.apply_rope(q), self.apply_rope(k)
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask[:, None, None, :], # (batch_size, 1, 1, time_steps)
            dropout_p=self.dropout if self.training else 0.0,
            scale=1.0 / math.sqrt(self.head_dim),
        ) # (batch_size, num_head, time_steps, head_dim = d_model // num_heads)
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, time_steps, self.num_heads * self.head_dim
        ) # (batch_size, time_steps, d_model)
        return self.output(attended) # (batch_size, time_steps, d_model)


class FeedForward(nn.Module):
    def __init__(self, config: ASRConfig) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.ffn_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ffn_dim, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape is (batch_size, time_steps, d_model)
        return self.network(x) # (batch_size, time_steps, d_model)


class ConformerConvolution(nn.Module):
    def __init__(self, config: ASRConfig) -> None:
        super().__init__()
        channels = config.d_model
        self.norm = nn.LayerNorm(channels)
        self.pointwise_in = nn.Conv1d(channels, 2 * channels, kernel_size=1) # (t+2*0-1)//1+1 = t
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=config.conv_kernel_size,
            padding=config.conv_kernel_size // 2,
            groups=channels,
        ) # (t+2*config.conv_kernel_size // 2-config.conv_kernel_size)//1+1 = t
        self.activation_norm = nn.LayerNorm(channels)
        self.pointwise_out = nn.Conv1d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x.shape is (batch_size, time_steps, d_model)
        # mask.shape is (batch_size, time_steps)
        x = self.norm(x).transpose(1, 2) # (batch_size, d_model, time_steps)
        x = x.masked_fill(~mask[:, None, :], 0.0)
        x = F.glu(self.pointwise_in(x), dim=1) # (batch_size, d_model, time_steps)
        # pointwise bias 会让 padding 重新非零，进入 depthwise 前必须再次 mask。
        x = x.masked_fill(~mask[:, None, :], 0.0)
        x = self.depthwise(x).transpose(1, 2) # (batch_size, time_steps, d_model)
        x = F.silu(self.activation_norm(x)).transpose(1, 2)
        x = self.dropout(self.pointwise_out(x))
        x = x.masked_fill(~mask[:, None, :], 0.0)
        return x.transpose(1, 2)


class ConformerBlock(nn.Module):
    def __init__(self, config: ASRConfig) -> None:
        super().__init__()
        self.ffn1 = FeedForward(config)
        self.ffn2 = FeedForward(config)
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.attention = RoPEMultiHeadAttention(config)
        self.attention_dropout = nn.Dropout(config.dropout)
        self.convolution = ConformerConvolution(config)
        self.output_norm = nn.LayerNorm(config.d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x.shape is (batch_size, time_steps, d_model)
        # mask.shape is (batch_size, time_steps)
        x = x + 0.5 * self.ffn1(x)
        x = x + self.attention_dropout(self.attention(self.attention_norm(x), mask))
        x = x + self.convolution(x, mask)
        x = x + 0.5 * self.ffn2(x)
        return self.output_norm(x)


class ConformerEncoder(nn.Module):
    def __init__(self, config: ASRConfig) -> None:
        super().__init__()
        self.subsampling = Conv2dSubsampling(config)
        self.blocks = nn.ModuleList(ConformerBlock(config) for _ in range(config.num_layers))

    def forward(
        self, features: torch.Tensor, frame_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # features.shape is (batch_size, num_window, num_mel)
        # frame_lengths.shape is (batch_size,)
        x, output_lengths, mask = self.subsampling(features, frame_lengths)
        for block in self.blocks:
            x = block(x, mask)
        return x.masked_fill(~mask[:, :, None], 0.0), output_lengths, mask
