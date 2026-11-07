from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ASRConfig:
    sample_rate: int = 16_000
    n_fft: int = 400
    win_length: int = 400
    hop_length: int = 160
    n_mels: int = 80
    f_min: float = 0.0
    f_max: float | None = None
    d_model: int = 64
    num_heads: int = 4
    num_layers: int = 2
    ffn_dim: int = 256
    conv_kernel_size: int = 9
    dropout: float = 0.1
    spec_augment: bool = False
    time_masks: int = 2
    freq_masks: int = 2
    max_time_mask: int = 20
    max_freq_mask: int = 12
    epsilon: float = 1e-5

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if not 0 < self.win_length <= self.n_fft or self.hop_length <= 0:
            raise ValueError("require 0 < win_length <= n_fft and positive hop_length")
        if self.n_mels < 8:
            raise ValueError("n_mels must be at least 8")
        f_max = self.sample_rate / 2 if self.f_max is None else self.f_max
        if not 0 <= self.f_min < f_max <= self.sample_rate / 2:
            raise ValueError("require 0 <= f_min < f_max <= sample_rate / 2")
        if self.d_model <= 0 or self.num_heads <= 0 or self.num_layers <= 0:
            raise ValueError("model dimensions and layer counts must be positive")
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if (self.d_model // self.num_heads) % 2:
            raise ValueError("per-head dimension must be even for RoPE")
        if self.ffn_dim <= 0 or self.conv_kernel_size < 3 or self.conv_kernel_size % 2 == 0:
            raise ValueError("ffn_dim must be positive and conv_kernel_size must be odd and >= 3")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if min(self.time_masks, self.freq_masks, self.max_time_mask, self.max_freq_mask) < 0:
            raise ValueError("SpecAugment parameters must be non-negative")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")

    @property
    def subsampled_mels(self) -> int:
        length = self.n_mels
        for _ in range(2):
            length = (length - 1) // 2
        if length < 1:
            raise ValueError("n_mels is too small for two convolutional subsampling layers")
        return length

    @property
    def subsampling_factor(self) -> int:
        return 4

    @property
    def output_frame_rate(self) -> float:
        return self.sample_rate / self.hop_length / self.subsampling_factor

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ASRConfig":
        return cls(**values)
