# Copyright (c) Kyutai, all rights reserved.
# Derived from kyutai-labs/moshi; semantic-distillation teaching path added.
# See NOTICE and LICENSES/MIT-moshi.txt.

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .config import MimiConfig
from .convolution import ConvDownsample1d, ConvTrUpsample1d, pad_for_conv1d
from .quantization import SplitResidualVectorQuantizer
from .seanet import SEANetDecoder, SEANetEncoder
from .streaming import StreamingContainer
from .transformer import ProjectedTransformer


@dataclass
class MimiOutput:
    reconstruction: torch.Tensor
    latent: torch.Tensor
    quantized: torch.Tensor
    codes: torch.Tensor
    commitment_loss: torch.Tensor
    bandwidth: torch.Tensor
    semantic: torch.Tensor
    semantic_prediction: torch.Tensor
    semantic_loss: torch.Tensor


class MimiModel(StreamingContainer):
    def __init__(self, config: MimiConfig | None = None) -> None:
        super().__init__()
        self.config = config or MimiConfig()
        self.encoder = SEANetEncoder(
            channels=self.config.channels,
            dimension=self.config.dimension,
            n_filters=self.config.n_filters,
            n_residual_layers=self.config.n_residual_layers,
            ratios=self.config.ratios,
        )
        self.decoder = SEANetDecoder(
            channels=self.config.channels, # default 1
            dimension=self.config.dimension, # default 64
            n_filters=self.config.n_filters, # default 16
            n_residual_layers=self.config.n_residual_layers, # default 1
            ratios=self.config.ratios, # default (8, 6, 5, 4)
        )
        transformer_kwargs = dict(
            input_dimension=self.config.dimension, # default 64
            output_dimensions=(self.config.dimension,), # default (64,)
            d_model=self.config.dimension, # default 64
            num_layers=self.config.transformer_num_layers, # default 2
            num_heads=self.config.transformer_num_heads, # default 4
            dim_feedforward=self.config.transformer_dim_feedforward, # default 128
            causal=True,
            context=self.config.transformer_context, # default 250
            positional_embedding="rope",
            max_period=10_000.0,
            norm="layer_norm",
            layer_scale=0.01,
            gating="none",
            conv_layout=True,
        )
        self.encoder_transformer = ProjectedTransformer(**transformer_kwargs)
        self.decoder_transformer = ProjectedTransformer(**transformer_kwargs)
        self.quantizer = SplitResidualVectorQuantizer(
            dimension=self.config.quantizer_dimension,
            input_dimension=self.config.dimension,
            output_dimension=self.config.dimension,
            n_q=self.config.num_codebooks,
            bins=self.config.codebook_size,
            decay=self.config.ema_decay,
            q_dropout=self.config.quantizer_dropout,
            no_quantization_rate=self.config.no_quantization_rate,
        )
        self.downsample = ConvDownsample1d(
            self.config.downsample_stride,
            dimension=self.config.dimension, # default 64
            causal=True,
            learnt=True,
        )
        self.upsample = ConvTrUpsample1d(
            self.config.downsample_stride,
            dimension=self.config.dimension,
            causal=True,
            learnt=True,
            channel_wise=True,
        )
        self.semantic_projection = nn.Conv1d(
            self.config.dimension, self.config.semantic_dimension, 1
        ) # (t+0-1)/1+1 = t 长度不变

    @property
    def sample_rate(self) -> int:
        return self.config.sample_rate

    @property
    def frame_rate(self) -> float:
        return self.config.frame_rate

    @property
    def frame_size(self) -> int:
        return self.config.frame_size

    @property
    def channels(self) -> int:
        return self.config.channels

    @property
    def num_codebooks(self) -> int:
        return self.quantizer.num_codebooks

    @property
    def total_codebooks(self) -> int:
        return self.quantizer.total_codebooks

    @property
    def cardinality(self) -> int:
        return self.quantizer.cardinality

    def set_num_codebooks(self, n: int) -> None:
        self.quantizer.set_num_codebooks(n)

    def freeze(self) -> None:
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def _check_waveform(self, x: torch.Tensor) -> None:
        # x.shape is (batch_size, 1, time_steps=sample_rate*seconds)
        if x.ndim != 3 or x.shape[1] != self.channels or x.shape[-1] == 0:
            raise ValueError(f"expected waveform [B, {self.channels}, T], got {tuple(x.shape)}")

    def _encode_to_unquantized_latent(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape is (batch_size, 1, time_steps=sample_rate*seconds)
        self._check_waveform(x)
        if self.is_streaming:
            if x.shape[-1] % self.frame_size:
                raise ValueError(
                    f"streaming input length must be a multiple of {self.frame_size}"
                )
        else:
            x = pad_for_conv1d(x, self.frame_size, self.frame_size)
        latent = self.encoder(x)
        latent = self.encoder_transformer(latent)[0]
        return self.downsample(latent)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        latent = self._encode_to_unquantized_latent(x)
        return self.quantizer.encode(latent)

    def encode_to_latent(self, x: torch.Tensor, quantize: bool = True) -> torch.Tensor:
        latent = self._encode_to_unquantized_latent(x)
        if not quantize:
            return latent
        return self.decode_latent(self.quantizer.encode(latent))

    def decode_latent(self, codes: torch.Tensor) -> torch.Tensor:
        return self.quantizer.decode(codes)

    def decode(self, codes: torch.Tensor, original_length: int | None = None) -> torch.Tensor:
        if codes.ndim != 3 or codes.shape[1] != self.num_codebooks:
            raise ValueError(f"expected codes [B, {self.num_codebooks}, F], got {tuple(codes.shape)}")
        latent = self.decode_latent(codes)
        latent = self.upsample(latent)
        latent = self.decoder_transformer(latent)[0]
        reconstruction = self.decoder(latent)
        if original_length is not None:
            if not 0 < original_length <= reconstruction.shape[-1]:
                raise ValueError("original_length is invalid for the padded reconstruction")
            reconstruction = reconstruction[..., :original_length]
        return reconstruction

    def forward(
        self, x: torch.Tensor, semantic_targets: torch.Tensor | None = None
    ) -> MimiOutput:
        # x.shape is (batch_size, 1, time_steps=sample_rate*seconds)
        if self.is_streaming:
            raise RuntimeError("use encode/decode for streaming; forward is the training path")
        original_length = x.shape[-1] # time_steps
        latent = self._encode_to_unquantized_latent(x) # (batch_size, input_dimension=64, num_frames=time_steps/1920)
        quantized = self.quantizer(latent, self.frame_rate) # quantized.shape is (batch_size, output_dimension, num_frames)
        reconstructed_latent = self.upsample(quantized.x) # upsample to (batch_size, output_dimension, num_frames * downsample_stride)
        reconstructed_latent = self.decoder_transformer(reconstructed_latent)[0]
        reconstruction = self.decoder(reconstructed_latent)[..., :original_length]
        if quantized.semantic is None:
            raise RuntimeError("Mimi split quantizer did not return its semantic branch")
        semantic_prediction = self.semantic_projection(quantized.semantic)
        semantic_loss = torch.zeros((), device=x.device) # 0D tensor
        if semantic_targets is not None:
            if semantic_targets.shape != semantic_prediction.shape:
                raise ValueError(
                    "semantic_targets must match [B, semantic_dimension, frames]: "
                    f"{tuple(semantic_prediction.shape)}"
                )
            semantic_loss = 1 - F.cosine_similarity(
                semantic_prediction, semantic_targets, dim=1
            ).mean() # （B,C,T) -> (B,T) -> () 0D tensor 只约束方向，不约束长度。余弦相似度对尺度不敏感,教师特征的绝对幅度没有意义（不同层的特征尺度不同），方向才编码语义。你只关心"这两个向量指向同一个语义方向"，不关心"学生输出的幅度是不是和教师一样"。如果用 MSE 做蒸馏，就变成了"幅度也要对齐"，反而引入了不必要的约束。
        return MimiOutput(
            reconstruction, # (batch_size, 1, time_steps=num_frames*1920)
            latent, # (batch_size, input_dimension=64, num_frames=time_steps/1920)
            quantized.x, # semantic.x + accoustic.x, (batch_size, output_dimension=64, num_frames=time_steps/1920)
            quantized.codes, # (batch_size, num_codebooks=8=1+M, num_frames=time_steps/1920)
            quantized.penalty, # 0D tensor, commitment loss
            quantized.bandwidth, # 0D tensor, bandwidth in bits per second
            quantized.semantic, # semantic.x, (batch_size, semantic_dimension=64, num_frames=time_steps/1920)
            semantic_prediction, # (batch_size, output_dimension, num_frames)
            semantic_loss, # 0D tensor, cosine similarity loss between semantic_prediction and semantic_targets
        )
