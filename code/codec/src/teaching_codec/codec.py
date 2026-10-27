from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .autoencoder import Decoder, Encoder
from .config import CodecConfig
from .quantization import ResidualVectorQuantizer


@dataclass
class CodecOutput:
    reconstruction: torch.Tensor
    latent: torch.Tensor
    quantized: torch.Tensor
    codes: torch.Tensor
    codebook_loss: torch.Tensor
    commitment_loss: torch.Tensor


class TeachingCodec(nn.Module):
    def __init__(self, config: CodecConfig | None = None) -> None:
        super().__init__()
        self.config = config or CodecConfig()
        self.encoder = Encoder(self.config)
        self.decoder = Decoder(self.config)
        self.quantizer = ResidualVectorQuantizer(
            self.config.codebook_size,
            self.config.latent_dim,
            self.config.num_codebooks,
        )

    def _check_waveform(self, x: torch.Tensor, require_hop_multiple: bool) -> None:
        if x.ndim != 3 or x.shape[1] != 1 or x.shape[2] == 0:
            raise ValueError(f"expected non-empty waveform [B, 1, T], got {tuple(x.shape)}")
        if require_hop_multiple and x.shape[-1] % self.config.hop_length:
            raise ValueError(f"T must be a multiple of hop_length={self.config.hop_length}")

    def forward(
        self,
        x: torch.Tensor,
        n_quantizers: int | None = None,
        quantizer_dropout: bool = False,
        dropout_rate: float = 1.0,
    ) -> CodecOutput:
        self._check_waveform(x, require_hop_multiple=True)
        latent = self.encoder(x)
        if self.config.use_quantization:
            result = self.quantizer(latent, n_quantizers, quantizer_dropout, dropout_rate)
            quantized = result.quantized
            codes = result.codes
            codebook_loss = result.codebook_loss
            commitment_loss = result.commitment_loss
        else:
            quantized = latent
            codes = torch.empty(
                x.shape[0], 0, latent.shape[-1], dtype=torch.long, device=x.device
            )
            codebook_loss = latent.sum() * 0
            commitment_loss = latent.sum() * 0
        reconstruction = self.decoder(quantized)
        if reconstruction.shape != x.shape:
            raise RuntimeError(f"round-trip shape changed: {tuple(x.shape)} -> {tuple(reconstruction.shape)}")
        return CodecOutput(
            reconstruction,
            latent,
            quantized,
            codes,
            codebook_loss,
            commitment_loss,
        )

    def _pad_to_hop(self, x: torch.Tensor) -> torch.Tensor:
        padding = (-x.shape[-1]) % self.config.hop_length
        return F.pad(x, (0, padding))

    def encode(self, x: torch.Tensor, n_quantizers: int | None = None) -> torch.Tensor:
        if not self.config.use_quantization:
            raise RuntimeError("encode/decode requires use_quantization=True")
        n_q = self.config.num_codebooks if n_quantizers is None else n_quantizers
        if not 1 <= n_q <= self.config.num_codebooks:
            raise ValueError(f"n_quantizers must be in [1, {self.config.num_codebooks}]")
        self._check_waveform(x, require_hop_multiple=False)
        latent = self.encoder(self._pad_to_hop(x))
        codes = []
        residual = latent
        for quantizer in self.quantizer.quantizers[:n_q]:
            result = quantizer(residual)
            codes.append(result.codes)
            residual = residual - result.quantized.detach()
        return torch.stack(codes, dim=1) # (batch_size,n_q,time_steps)

    def decode(self, codes: torch.Tensor, original_length: int | None = None) -> torch.Tensor:
        if not self.config.use_quantization:
            raise RuntimeError("encode/decode requires use_quantization=True")
        quantized = self.quantizer.decode(codes) # (batch_size, latent_dim = 32, time_steps)
        reconstruction = self.decoder(quantized) # (batch_size,final_channels=1,time_steps)
        if original_length is not None:
            if not 0 < original_length <= reconstruction.shape[-1]:
                raise ValueError(
                    f"original_length={original_length} is invalid for padded length {reconstruction.shape[-1]}"
                )
            reconstruction = reconstruction[..., :original_length]
        return reconstruction

    def reconstruct(self, x: torch.Tensor, n_quantizers: int | None = None) -> torch.Tensor:
        original_length = x.shape[-1]
        return self.decode(self.encode(x, n_quantizers), original_length)
