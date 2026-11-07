from dataclasses import asdict, dataclass, field
from typing import Any

from .config import ASRConfig


@dataclass
class AudioConfig:
    sample_rate: int = 16_000
    n_fft: int = 400
    win_length: int = 400
    hop_length: int = 160
    n_mels: int = 128
    d_model: int = 64
    encoder_layers: int = 2
    encoder_heads: int = 4
    ffn_dim: int = 256
    downsample_channels: int = 32
    attention_window: int = 25
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.n_mels <= 0 or self.d_model <= 0 or self.encoder_heads <= 0:
            raise ValueError("audio dimensions must be positive")
        if self.d_model % self.encoder_heads:
            raise ValueError("d_model must be divisible by encoder_heads")
        if min(self.encoder_layers, self.ffn_dim, self.downsample_channels) < 1:
            raise ValueError("audio layer and channel counts must be positive")
        if self.attention_window < 0 or not 0 <= self.dropout < 1:
            raise ValueError("invalid attention window or dropout")

    @property
    def output_frame_rate(self) -> float:
        return self.sample_rate / self.hop_length / 8

    @property
    def downsampled_mels(self) -> int:
        length = self.n_mels
        for _ in range(3):
            length = (length + 1) // 2
        return length

    def feature_config(self) -> ASRConfig:
        return ASRConfig(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            dropout=self.dropout,
        )


@dataclass
class LLMConfig:
    vocab_size: int = 33
    hidden_size: int = 64
    intermediate_size: int = 128
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    head_dim: int = 16
    rope_theta: float = 5_000_000.0
    rms_norm_eps: float = 1e-6
    attention_dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.vocab_size <= 0 or self.hidden_size <= 0 or self.intermediate_size <= 0:
            raise ValueError("LLM dimensions must be positive")
        if self.num_hidden_layers <= 0 or self.num_attention_heads <= 0:
            raise ValueError("LLM layer and head counts must be positive")
        if self.num_key_value_heads <= 0 or self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("attention heads must be divisible by key/value heads")
        if self.head_dim <= 0 or self.head_dim % 2:
            raise ValueError("head_dim must be positive and even for RoPE")
        if self.rope_theta <= 0 or self.rms_norm_eps <= 0:
            raise ValueError("RoPE theta and RMSNorm epsilon must be positive")
        if not 0 <= self.attention_dropout < 1:
            raise ValueError("attention_dropout must be in [0, 1)")


@dataclass
class ModernASRConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ModernASRConfig":
        return cls(AudioConfig(**values["audio"]), LLMConfig(**values["llm"]))
