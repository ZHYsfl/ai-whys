from dataclasses import asdict, dataclass
from math import log2, prod
from typing import Any


@dataclass
class CodecConfig:
    sample_rate: int = 16_000
    latent_dim: int = 32
    base_channels: int = 32
    strides: tuple[int, ...] = (2, 4, 4, 4)
    codebook_size: int = 64
    num_codebooks: int = 4
    codebook_weight: float = 1.0
    commitment_weight: float = 0.25
    use_quantization: bool = True

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.latent_dim <= 0 or self.base_channels <= 0:
            raise ValueError("model dimensions must be positive")
        if not self.strides or any(stride < 2 or stride % 2 for stride in self.strides):
            raise ValueError("teaching strides must be even integers >= 2")
        if self.codebook_size < 2 or self.num_codebooks < 1:
            raise ValueError("need at least two codes and one codebook")

    @property
    def hop_length(self) -> int:
        return prod(self.strides)

    @property
    def frame_rate(self) -> float:
        return self.sample_rate / self.hop_length

    def bitrate(self, num_codebooks: int | None = None) -> float:
        if not self.use_quantization:
            return 0.0
        n_q = self.num_codebooks if num_codebooks is None else num_codebooks
        if not 1 <= n_q <= self.num_codebooks:
            raise ValueError(f"num_codebooks must be in [1, {self.num_codebooks}]")
        return self.frame_rate * n_q * log2(self.codebook_size)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "CodecConfig":
        values = dict(values)
        values["strides"] = tuple(values["strides"])
        return cls(**values)
