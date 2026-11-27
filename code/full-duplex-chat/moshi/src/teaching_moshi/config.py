from dataclasses import asdict, dataclass
from math import prod
from typing import Any


@dataclass
class MimiConfig:
    sample_rate: int = 24_000
    channels: int = 1
    frame_rate: float = 12.5
    dimension: int = 64
    n_filters: int = 16
    n_residual_layers: int = 1
    ratios: tuple[int, ...] = (8, 6, 5, 4)
    quantizer_dimension: int = 64
    num_codebooks: int = 8
    codebook_size: int = 64
    transformer_num_layers: int = 2
    transformer_num_heads: int = 4
    transformer_dim_feedforward: int = 128
    transformer_context: int = 250
    semantic_dimension: int = 64
    quantizer_dropout: bool = True
    no_quantization_rate: float = 0.5
    ema_decay: float = 0.99

    def __post_init__(self) -> None:
        if self.channels != 1 or self.sample_rate <= 0 or self.frame_rate <= 0:
            raise ValueError("Mimi expects mono audio and positive sample/frame rates")
        if not self.ratios or any(ratio < 2 for ratio in self.ratios):
            raise ValueError("all SeaNet ratios must be at least 2")
        if self.dimension % self.transformer_num_heads:
            raise ValueError("dimension must be divisible by transformer_num_heads")
        if self.num_codebooks < 2 or self.codebook_size < 2:
            raise ValueError("split RVQ needs at least two codebooks and two codes")
        if not 0 <= self.no_quantization_rate <= 1:
            raise ValueError("no_quantization_rate must be in [0, 1]")
        if self.encoder_frame_rate % self.frame_rate:
            raise ValueError("encoder_frame_rate must be an integer multiple of frame_rate")

    @property
    def hop_length(self) -> int:
        return prod(self.ratios)

    @property
    def encoder_frame_rate(self) -> float:
        return self.sample_rate / self.hop_length

    @property
    def downsample_stride(self) -> int:
        return int(self.encoder_frame_rate / self.frame_rate)

    @property
    def frame_size(self) -> int:
        return int(self.sample_rate / self.frame_rate) # default 24000/12.5 = 1920 samples per frame

    @classmethod
    def official(cls) -> "MimiConfig":
        return cls(
            dimension=512,
            n_filters=64,
            quantizer_dimension=512,
            num_codebooks=32,
            codebook_size=2048,
            transformer_num_layers=8,
            transformer_num_heads=8,
            transformer_dim_feedforward=2048,
            semantic_dimension=1024,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "MimiConfig":
        values = dict(values)
        values["ratios"] = tuple(values["ratios"])
        return cls(**values)


@dataclass
class MoshiConfig:
    n_q: int = 16
    dep_q: int = 8
    card: int = 64
    text_card: int = 260
    dim: int = 64
    num_heads: int = 4
    num_layers: int = 2
    hidden_scale: float = 4.0
    context: int = 3000
    max_period: float = 10_000.0
    depformer_dim: int = 32
    depformer_dim_feedforward: int = 128
    depformer_num_heads: int = 4
    depformer_num_layers: int = 2
    depformer_norm: str | None = None
    depformer_context: int = 8
    depformer_weights_per_step: bool = True
    depformer_multi_linear: bool = True
    depformer_weights_per_step_schedule: tuple[int, ...] | None = None
    delays: tuple[int, ...] = (0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1)
    existing_text_padding_id: int = 3
    existing_text_end_padding_id: int = 0

    def __post_init__(self) -> None:
        if self.n_q != 2 * self.dep_q:
            raise ValueError("full-duplex Moshi needs equally many Moshi and user audio streams")
        if self.dep_q < 2 or self.card < 2 or self.text_card < 4:
            raise ValueError("invalid audio/text vocabulary or codebook count")
        if self.dim % self.num_heads or self.depformer_dim % self.depformer_num_heads:
            raise ValueError("model dimensions must be divisible by attention heads")
        if (self.dim // self.num_heads) % 2:
            raise ValueError("RoPE requires an even temporal head dimension")
        if len(self.delays) != self.num_codebooks or any(delay < 0 for delay in self.delays):
            raise ValueError(f"delays must contain {self.num_codebooks} non-negative entries")
        if self.depformer_weights_per_step_schedule is not None:
            if len(self.depformer_weights_per_step_schedule) != self.dep_q:
                raise ValueError("the per-step schedule must have dep_q entries")
        if not 0 <= self.existing_text_end_padding_id < self.text_card:
            raise ValueError("invalid EPAD token")
        if not 0 <= self.existing_text_padding_id < self.text_card:
            raise ValueError("invalid PAD token")

    @property
    def num_codebooks(self) -> int:
        return self.n_q + 1

    @property
    def audio_offset(self) -> int:
        return 1

    @property
    def user_audio_offset(self) -> int:
        return self.dep_q + 1

    @property
    def user_codebooks(self) -> int:
        return self.n_q - self.dep_q

    @property
    def max_delay(self) -> int:
        return max(self.delays)

    @classmethod
    def official(cls) -> "MoshiConfig":
        return cls(
            card=2048,
            text_card=32_000,
            dim=4096,
            num_heads=32,
            num_layers=32,
            hidden_scale=4.125,
            depformer_dim=1024,
            depformer_dim_feedforward=4224,
            depformer_num_heads=16,
            depformer_num_layers=6,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "MoshiConfig":
        values = dict(values)
        values["delays"] = tuple(values["delays"])
        if values.get("depformer_weights_per_step_schedule") is not None:
            values["depformer_weights_per_step_schedule"] = tuple(
                values["depformer_weights_per_step_schedule"]
            )
        return cls(**values)
