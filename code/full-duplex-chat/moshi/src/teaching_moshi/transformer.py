# Copyright (c) Kyutai, all rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Derived from kyutai-labs/moshi and AudioCraft; see NOTICE and LICENSES/.

import math
from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from .streaming import State, StreamingContainer, StreamingModule


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8, dtype: torch.dtype | None = torch.float32) -> None:
        super().__init__()
        self.eps = eps
        self.dtype = dtype
        self.alpha = nn.Parameter(torch.ones(1, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape is (b, t, d)
        input_dtype = x.dtype
        if self.dtype is not None:
            x = x.to(self.dtype)
        variance = self.eps + x.square().mean(dim=-1, keepdim=True) # (b, t, 1)
        return (x * (self.alpha.to(variance) * torch.rsqrt(variance))).to(input_dtype)


class LayerScale(nn.Module):
    def __init__(self, channels: int, init: float) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.full((channels,), init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape is (b, t, d)
        return self.scale * x # (d) * (b, t, d) -> (b, t, d)


def create_norm_fn(norm: str, dim: int) -> nn.Module:
    if norm == "layer_norm":
        return nn.LayerNorm(dim, eps=1e-5)
    if norm == "rms_norm_f32": # dtype 默认 torch.float32，eps 默认 1e-8    
        return RMSNorm(dim)
    if norm == "rms_norm":
        return RMSNorm(dim, eps=1e-5, dtype=None)
    raise ValueError(f"unknown norm: {norm}")


def apply_rope(
    q: torch.Tensor, # (B,H,T,D)
    k: torch.Tensor, # (B,H,T,D)
    offset: torch.Tensor, # (B,)
    max_period: float = 10_000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    # q/k: [B, H, T, D]; adjacent pairs (0,1), (2,3), ... rotate together.
    batch_size, _, time_steps, head_dim = q.shape
    if head_dim % 2:
        raise ValueError("RoPE head dimension must be even")
    frequencies = torch.exp(
        torch.arange(head_dim // 2, device=q.device, dtype=torch.float32) # (D/2，)
        * (-math.log(max_period) * 2 / head_dim)
    ) # (D/2,)
    positions = offset.float()[:, None] + torch.arange(time_steps, device=q.device) # （B, T）
    angles = positions[:, None, :, None] * frequencies[None, None, None, :] # (B, 1, T, D/2)
    cos, sin = angles.cos(), angles.sin()

    def rotate(x: torch.Tensor) -> torch.Tensor:
        # (B,H,T,D)
        input_dtype = x.dtype
        pairs = x.float().view(batch_size, x.shape[1], time_steps, head_dim // 2, 2)
        real, imaginary = pairs[..., 0], pairs[..., 1]
        rotated = torch.stack(
            (real * cos - imaginary * sin, real * sin + imaginary * cos), dim=-1
        )
        return rotated.flatten(-2).to(input_dtype)

    return rotate(q), rotate(k)


class RotaryEmbedding(nn.Module):
    def __init__(self, max_period: float = 10_000.0) -> None:
        super().__init__()
        self.max_period = max_period

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, offset: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return apply_rope(q, k, offset, self.max_period)


class KVCacheResult(NamedTuple):
    keys: torch.Tensor
    values: torch.Tensor
    positions: torch.Tensor


class RingKVCache:
    """Fixed-capacity K/V storage: [2, B, H, context, D]."""

    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        capacity: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if capacity < 1:
            raise ValueError("KV cache capacity must be positive")
        self.capacity = capacity
        self.cache = torch.zeros(
            2, batch_size, num_heads, capacity, head_dim, device=device, dtype=dtype
        )
        self.end_offset = torch.zeros(batch_size, device=device, dtype=torch.long)

    def reset(self) -> None:
        self.cache.zero_()
        self.end_offset.zero_()

    def complete(self, k: torch.Tensor, v: torch.Tensor) -> KVCacheResult:
        # k.shape is (B,H,T,D); v.shape is (B,H,T,D)
        batch_size, _, time_steps, _ = k.shape
        indexes = (
            torch.arange(time_steps, device=k.device)[None]
            + self.end_offset[:, None]
        ) % self.capacity # （B, T) indexes[b, t] = 第 b 条流、这次第 t 帧应该写到的物理槽
        scatter_indexes = indexes[:, None, :, None].expand(-1, k.shape[1], -1, k.shape[-1]) # （B, H, T, D) scatter_indexes[b, h, t, d] = 第 b 条流、头 h、这次第 t 帧、维度 d 应该写到的物理槽
        self.cache[0].scatter_(2, scatter_indexes, k)
        self.cache[1].scatter_(2, scatter_indexes, v)

        last_offset = self.end_offset[:, None] + time_steps - 1 # (B,1) 这次写入的最后一帧，它的绝对位置是多少
        ring_positions = torch.arange(self.capacity, device=k.device)[None] # （1, capacity） 物理槽的绝对位置是多少
        delta = ring_positions - last_offset % self.capacity # （B, capacity） delta[b, r] = 物理槽 r 相对最新槽 w 的偏移
        positions = torch.where(
            delta <= 0, last_offset + delta, last_offset + delta - self.capacity
        ) # 每个物理槽对应的绝对位置
        self.end_offset += time_steps
        valid = ring_positions < self.end_offset[:, None]
        positions = torch.where(valid, positions, torch.full_like(positions, -1)) # （B, capacity） 物理槽 r 对应的绝对位置，如果这个槽还没有写过数据，就标记为 -1
        return KVCacheResult(self.cache[0], self.cache[1], positions)


def apply_weights_per_step(
    modules: nn.ModuleList,
    schedule: tuple[int, ...] | None,
    x: torch.Tensor,
    offset: int,
) -> torch.Tensor:
    # Depth step k uses its own weights; Temporal Transformer normally has one shared set.
    if len(modules) == 1:
        return modules[0](x)
    
    outputs = []
    for time_index in range(x.shape[1]):
        module_index = offset + time_index
        if schedule is not None:
            module_index = schedule[module_index]
        outputs.append(modules[module_index](x[:, time_index : time_index + 1]))
    return torch.cat(outputs, dim=1)


@dataclass
class AttentionState(State):
    kv_cache: RingKVCache | None = None
    offset: torch.Tensor | None = None
    offset_cpu: int = 0

    def reset(self) -> None:
        if self.kv_cache is not None:
            self.kv_cache.reset()
        if self.offset is not None:
            self.offset.zero_()
        self.offset_cpu = 0


class StreamingMultiheadAttention(StreamingModule):
    def __init__(
        self,
        embed_dim: int, # default 64
        num_heads: int, # default 4
        causal: bool, # default True
        context: int | None, # default 250
        rope: RotaryEmbedding | None,
        weights_per_step: int = 0, # default 0
        weights_per_step_schedule: tuple[int, ...] | None = None, # default None
    ) -> None:
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.causal = causal
        self.context = context
        self.rope = rope
        self.weights_per_step = weights_per_step
        self.weights_per_step_schedule = weights_per_step_schedule
        count = weights_per_step or 1
        if weights_per_step_schedule is not None: # 若为 None,没有 schedule ,默认就是每个位置一套
            count = max(weights_per_step_schedule) + 1 # 假设：weights_per_step_schedule = (0, 1, 1, 2) 那么实际只需要：组 0、组 1、组 2 一共 3 套 
        self.in_projs = nn.ModuleList(
            [nn.Linear(embed_dim, 3 * embed_dim, bias=False) for _ in range(count)]
        )
        self.out_projs = nn.ModuleList(
            [nn.Linear(embed_dim, embed_dim, bias=False) for _ in range(count)]
        )

    def _init_streaming_state(self, batch_size: int) -> AttentionState:
        parameter = self.in_projs[0].weight
        capacity = self.context or self.weights_per_step
        if not capacity:
            raise RuntimeError("streaming attention needs a finite context")
        cache = RingKVCache(
            batch_size,
            self.num_heads,
            self.embed_dim // self.num_heads,
            capacity,
            parameter.device,
            parameter.dtype,
        )
        return AttentionState(
            batch_size,
            parameter.device,
            cache,
            torch.zeros(batch_size, device=parameter.device, dtype=torch.long),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape is (batch_size,time_steps/stride/960,dimension = 64)
        batch_size, time_steps, _ = x.shape
        state: AttentionState | None = self._streaming_state
        if state is not None and time_steps > 1: # 当前有挂载流式状态且time_steps>1,递归
            return torch.cat(
                [self.forward(x[:, index : index + 1]) for index in range(time_steps)],
                dim=1,
            )
        if state is None: # 当前没有挂载流式状态(训练)
            offset = torch.zeros(batch_size, device=x.device, dtype=torch.long)
            offset_cpu = 0
        else: # 当前有挂载流式状态且time_steps=1
            if state.offset is None:
                raise RuntimeError("incomplete attention state")
            offset = state.offset
            offset_cpu = state.offset_cpu

        projected = apply_weights_per_step(
            self.in_projs, self.weights_per_step_schedule, x, offset_cpu
        ) # 就是个 transformer 的 QKV 线性变换，x.shape is (batch_size,time_steps/stride/960,dimension = 64) -> projected.shape is (batch_size,time_steps/stride/960,dimension = 64*3)
        qkv = projected.view(
            batch_size, time_steps, 3, self.num_heads, self.embed_dim // self.num_heads
        ).permute(2, 0, 3, 1, 4) # qkv.shape is (3,batch_size,num_heads,time_steps/stride/960,dimension = 64/num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.rope is not None:
            q, k = self.rope(q, k, offset)

        if state is None: # 当前没有挂载流式状态
            keys, values = k, v
            key_positions = torch.arange(time_steps, device=x.device)[None].expand(
                batch_size, -1
            ) # (B, T)
        else:
            # 当前有挂载流式状态
            if state.kv_cache is None:
                raise RuntimeError("missing KV cache")
            keys, values, key_positions = state.kv_cache.complete(k, v) # key_positions.shape is (B, capacity) key_positions[b, r] = 第 b 条流、物理槽 r 对应的绝对位置，如果这个槽还没有写过数据，就标记为 -1

        query_positions = offset[:, None, None] + torch.arange(
            time_steps, device=x.device
        ).view(1, -1, 1) # (B, T, 1) query_positions[b, t, 0] = 第 b 条流、当前 chunk 第 t 个 query 的绝对位置
        delta = query_positions - key_positions[:, None, :] # （B, T, T/capacity） delta[b, t, r] = 第 b 条流 第 t 个 query 的绝对位置 - 第 l 个 key 的绝对位置
        # delta > 0：key 在 query 之前，是过去
        # delta = 0：key 就是 query 自己
        # delta < 0：key 在 query 之后，是未来
        allowed = (key_positions[:, None, :] >= 0) & (delta >= 0) # （B, T, T/capacity）
        if self.context is not None:
            allowed &= delta < self.context # （B, T, T/capacity） allowed[b, t, r] = 第 b 条流 第 t 个 query 是否允许访问第 r 个 key
        attended = F.scaled_dot_product_attention(
            q, keys, values, attn_mask=allowed[:, None], dropout_p=0.0
        ) # attended.shape is (batch_size,num_heads,time_steps/stride/960,dimension = 64/num_heads)
        attended = attended.transpose(1, 2).reshape(batch_size, time_steps, self.embed_dim)
        output = apply_weights_per_step(
            self.out_projs, self.weights_per_step_schedule, attended, offset_cpu
        )
        if state is not None: # 当前有挂载流式状态
            state.offset = offset + time_steps
            state.offset_cpu += time_steps
        return output


class ActivationGating(nn.Module):
    """SwiGLU-family FFN: Linear -> split -> activation(gate) * value -> Linear."""

    def __init__(self, dim: int, dim_feedforward: int, activation=F.silu) -> None:
        super().__init__()
        hidden = (21 * dim) // 8 if dim_feedforward == 4 * dim else (2 * dim_feedforward) // 3
        self.linear_in = nn.Linear(dim, 2 * hidden, bias=False)
        self.linear_out = nn.Linear(hidden, dim, bias=False)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, _ = x.shape
        gate, value = self.linear_in(x).view(batch_size, time_steps, 2, -1).unbind(dim=2)
        return self.linear_out(self.activation(gate) * value)


@dataclass
class LayerState(State):
    offset_cpu: int = 0

    def reset(self) -> None:
        self.offset_cpu = 0


class StreamingTransformerLayer(StreamingModule):
    def __init__(
        self,
        d_model: int, # default 64
        num_heads: int, # default 4
        dim_feedforward: int, # default 128
        causal: bool, # default True
        context: int | None, # default 250
        rope: RotaryEmbedding | None,
        norm: str, # default "layernorm"
        layer_scale: float | None, # default 0.01
        gating: str, # default "none"
        weights_per_step: int = 0,
        weights_per_step_schedule: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        self.self_attn = StreamingMultiheadAttention(
            d_model, # default 64
            num_heads, # default 4
            causal, # default True
            context, # default 250
            rope,
            weights_per_step, # default 0
            weights_per_step_schedule, # default None
        )
        self.norm1 = create_norm_fn(norm, d_model)
        self.norm2 = create_norm_fn(norm, d_model)
        self.weights_per_step = weights_per_step
        self.weights_per_step_schedule = weights_per_step_schedule
        count = weights_per_step or 1
        if weights_per_step_schedule is not None:
            count = max(weights_per_step_schedule) + 1
        if gating == "silu":
            self.gating = (
                nn.ModuleList(
                    [ActivationGating(d_model, dim_feedforward) for _ in range(count)]
                )
                if weights_per_step
                else ActivationGating(d_model, dim_feedforward)
            )
            self.linear1 = None
            self.linear2 = None
        elif gating == "none":
            if weights_per_step:
                raise ValueError("per-step FFN requires gated FFN in the Moshi architecture")
            self.gating = None
            self.linear1 = nn.Linear(d_model, dim_feedforward, bias=False)
            self.linear2 = nn.Linear(dim_feedforward, d_model, bias=False)
        else:
            raise ValueError(f"unknown gating: {gating}")
        self.layer_scale_1 = (
            nn.Identity() if layer_scale is None else LayerScale(d_model, layer_scale)
        )
        self.layer_scale_2 = (
            nn.Identity() if layer_scale is None else LayerScale(d_model, layer_scale)
        )

    def _init_streaming_state(self, batch_size: int) -> LayerState:
        device = self.norm1.weight.device if hasattr(self.norm1, "weight") else self.norm1.alpha.device
        return LayerState(batch_size, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape is (batch_size,time_steps/stride/960,dimension = 64)
        x = x + self.layer_scale_1(self.self_attn(self.norm1(x)))
        normalized = self.norm2(x)
        if self.gating is None:
            update = self.linear2(F.gelu(self.linear1(normalized)))
        else:
            state: LayerState | None = self._streaming_state
            offset = 0 if state is None else state.offset_cpu
            update = (
                apply_weights_per_step(
                    self.gating, self.weights_per_step_schedule, normalized, offset
                )
                if self.weights_per_step
                else self.gating(normalized)
            )
        x = x + self.layer_scale_2(update)
        state = self._streaming_state
        if state is not None:
            state.offset_cpu += x.shape[1]
        return x


class StreamingTransformer(StreamingModule):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int,
        causal: bool = True,
        context: int | None = None,
        positional_embedding: str = "rope",
        max_period: float = 10_000.0,
        norm: str = "rms_norm_f32",
        layer_scale: float | None = None,
        gating: str = "silu",
        weights_per_step: int = 0,
        weights_per_step_schedule: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        if positional_embedding not in {"rope", "none"}:
            raise ValueError("Moshi/Mimi use RoPE or no positional embedding")
        self.context = context # default 250
        self.rope = RotaryEmbedding(max_period) if positional_embedding == "rope" else None
        self.layers = nn.ModuleList(
            [
                StreamingTransformerLayer(
                    d_model,
                    num_heads,
                    dim_feedforward,
                    causal,
                    context,
                    self.rope,
                    norm,
                    layer_scale,
                    gating,
                    weights_per_step,
                    weights_per_step_schedule,
                )
                for _ in range(num_layers)
            ]
        )

    def _init_streaming_state(self, batch_size: int) -> State:
        parameter = next(parameter for parameter in self.parameters())
        return State(batch_size, parameter.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class ProjectedTransformer(StreamingContainer):
    """Adapter between convolution layout [B,C,T] and Transformer layout [B,T,C]."""

    def __init__(
        self,
        input_dimension: int,
        output_dimensions: tuple[int, ...],
        d_model: int,
        conv_layout: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.transformer = StreamingTransformer(d_model=d_model, **kwargs)
        self.input_proj = (
            nn.Identity()
            if input_dimension == d_model
            else nn.Linear(input_dimension, d_model, bias=False)
        )
        self.output_projs = nn.ModuleList(
            [
                nn.Identity() if output_dimension == d_model else nn.Linear(d_model, output_dimension)
                for output_dimension in output_dimensions
            ]
        )
        self.conv_layout = conv_layout

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        # x.shape is (batch_size,dimension = 64,time_steps/stride/960)
        if self.conv_layout:
            x = x.transpose(1, 2) # x.shape is (batch_size,time_steps/stride/960,dimension = 64)
        x = self.input_proj(x)
        x = self.transformer(x)
        outputs = []
        for projection in self.output_projs:
            y = projection(x)
            if self.conv_layout:
                y = y.transpose(1, 2)
            outputs.append(y)
        return outputs
