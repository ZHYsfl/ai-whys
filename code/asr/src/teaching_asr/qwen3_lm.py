from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .modern_config import LLMConfig


class RMSNorm(nn.Module):
    '''
    root mean square layer normalization 均方根归一化

    LayerNorm ：先减均值，再除以标准差，通常还有 weight 和 bias
    RMSNorm ：不减均值，只除以均方根，通常只有 weight
    RMSNorm 少做了均值中心化，计算更简洁，是现代 LLM 中很常见的归一化方式，但它并不是在所有任务上都必然优于 LayerNorm
    '''
    def __init__(self, hidden_size: int, epsilon: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size)) # (hidden_size,)
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape is (batch, time_steps, hidden_size)
        variance = x.float().square().mean(dim=-1, keepdim=True) # variance.shape is (batch, time_steps, 1)
        return self.weight * (x.float() * torch.rsqrt(variance + self.epsilon)).to(x.dtype) # rsqrt = 1/ sqrt 另外，广播时从最后一个维度开始对齐


class SwiGLU(nn.Module):
    def __init__(self, config: LLMConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape is (batch_size, time_steps, hidden_size)
        return self.down(F.silu(self.gate(x)) * self.up(x))


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    # x.shape is (batch_size, num_head, time_steps, head_dim)
    first, second = x.chunk(2, dim=-1)
    # first.shape is (batch_size, num_head, time_steps, head_dim/2)
    # second.shape is (batch_size, num_head, time_steps, head_dim/2)

    return torch.cat((-second, first), dim=-1) # shape is (batch_size, num_head, time_steps, head_dim)


def rotary_embeddings(
    positions: torch.Tensor, # (time_steps,) or (batch_size, time_steps)
    head_dim: int, # default 16
    theta: float, # default 5_000_000.0
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=positions.device).float() / head_dim)
    ) # inv_freq.shape is (head_dim/2,)


    angles = positions.float()[..., None] * inv_freq
    # positions.float()[...,None].shape is (time_steps, 1) or (batch_size, time_steps, 1)
    # inv_freq.shape is (head_dim/2,)
    # angles.shape is (time_steps, head_dim/2) or (batch_size, time_steps, head_dim/2)


    angles = torch.cat((angles, angles), dim=-1)
    # (time_steps, head_dim) or (batch_size, time_steps, head_dim)
    # x0 x8 使用 theta0
    # x1 x9 使用 theta1
    # ...
    # x7 x15 使用 theta7

    if angles.ndim == 2:
        angles = angles[None] # (1, time_steps, head_dim)
    return angles.cos()[:, None].to(dtype), angles.sin()[:, None].to(dtype)
    # angles.cos()[:, None].to(dtype).shape is (1, 1, time_steps, head_dim) or (batch_size, 1, time_steps, head_dim)
    # angles.sin()[:, None].to(dtype).shape is (1, 1, time_steps, head_dim) or (batch_size, 1, time_steps, head_dim)


def repeat_kv(x: torch.Tensor, repeats: int) -> torch.Tensor:
    # x.shape is (batch_size, num_kv_head, time_steps, head_dim)
    batch, kv_heads, time_steps, head_dim = x.shape
    if repeats == 1:
        return x
    return x[:, :, None].expand(batch, kv_heads, repeats, time_steps, head_dim).reshape(
        batch, kv_heads * repeats, time_steps, head_dim
    ) # (batch_size, num_kv_head * repeats = num_q_head, time_steps, head_dim)


class Qwen3Attention(nn.Module):
    def __init__(self, config: LLMConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads # default 4
        self.num_kv_heads = config.num_key_value_heads # default 2
        self.head_dim = config.head_dim # default 16
        self.repeats = self.num_heads // self.num_kv_heads # default 2
        self.dropout = config.attention_dropout # default 0.0

        # 普通 MHA 通常让 Q/K/V 都有 4 个头；这里 GQA 为了节省 K/V 的计算和缓存，只给 K/V 各 2 个头。
        # 补充知识：
        # MQA 和 GQA 仍然可以按普通 MHA 理解：每个 Q 头分别做 Attention，最后 concat
        # MQA：所有 Q 头共享同一套 K/V
        # GQA：组内 Q 头共享一套 K/V，不同组使用不同 K/V
        # 前向时共享 K/V 的权重、向量和缓存
        # 反向时，各使用方传来的梯度会累加到同一套共享 K/V 参数上，再统一更新

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)

        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor, # (batch_size, time_steps = max_length=max_across_batch(prompt_length+speech_length+target_length), hidden_size)
        cos: torch.Tensor, # (1, 1, time_steps, head_dim) or (batch_size, 1, time_steps, head_dim)
        sin: torch.Tensor, # (1, 1, time_steps, head_dim) or (batch_size, 1, time_steps, head_dim)
        attention_mask: torch.Tensor, # (batch_size, 1, time_steps, past_length + time_steps)
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None, # a tuple of (past_k, past_v) where past_k.shape is (batch_size, num_kv_head, past_length, head_dim) and past_v.shape is (batch_size, num_kv_head, past_length, head_dim)
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        batch_size, time_steps, _ = x.shape
        q = self.q_norm(self.q_proj(x).view(batch_size, time_steps, self.num_heads, self.head_dim))
        k = self.k_norm(self.k_proj(x).view(batch_size, time_steps, self.num_kv_heads, self.head_dim))
        v = self.v_proj(x).view(batch_size, time_steps, self.num_kv_heads, self.head_dim)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        # q.shape is (batch_size, num_q_head, time_steps, head_dim)
        # k.shape is (batch_size, num_kv_head, time_steps, head_dim)
        # v.shape is (batch_size, num_kv_head, time_steps, head_dim)
        # Q 有 H 个头，K/V 只有 KV 个头；attention 前再把 K/V 展开到 H 个头。
        q = q * cos + rotate_half(q) * sin # (batch_size, num_q_head, time_steps, head_dim)
        k = k * cos + rotate_half(k) * sin # (batch_size, num_kv_head, time_steps, head_dim)
        if past_key_value is not None:
            past_k, past_v = past_key_value # past_k or past_v.shape is (batch_size, num_kv_head, past_length, head_dim)
            k = torch.cat((past_k, k), dim=-2) # (batch_size, num_kv_head, past_length + time_steps, head_dim)
            v = torch.cat((past_v, v), dim=-2) # (batch_size, num_kv_head, past_length + time_steps, head_dim)
        present = (k, v)
        attended = F.scaled_dot_product_attention(
            q,
            repeat_kv(k, self.repeats),
            repeat_kv(v, self.repeats),
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            scale=self.head_dim**-0.5,
        ) # attended.shape is (batch_size, num_q_head, time_steps, head_dim)
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, time_steps, self.num_heads * self.head_dim
        ) # (batch_size, time_steps, hidden_size) where hidden_size = num_q_head * head_dim
        return self.o_proj(attended), present


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: LLMConfig) -> None:
        super().__init__()
        self.input_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attention = Qwen3Attention(config)
        self.post_attention_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = SwiGLU(config)

    def forward(
        self,
        x: torch.Tensor, # (batch_size, max_length=max_across_batch(prompt_length+speech_length+target_length), hidden_size)
        cos: torch.Tensor, # (1, 1, time_steps, head_dim) or (batch_size, 1, time_steps, head_dim)
        sin: torch.Tensor, # (1, 1, time_steps, head_dim) or (batch_size, 1, time_steps, head_dim)
        attention_mask: torch.Tensor, # (batch_size, 1, time_steps, past_length + time_steps)
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None, # length is num_hidden_layers, each element is a tuple of (past_k, past_v) where past_k.shape is (batch_size, num_kv_head, past_length, head_dim) and past_v.shape is (batch_size, num_kv_head, past_length, head_dim)
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        attended, present = self.attention(
            self.input_norm(x), cos, sin, attention_mask, past_key_value
        )
        x = x + attended
        return x + self.mlp(self.post_attention_norm(x)), present


@dataclass
class QwenLMOutput:
    logits: torch.Tensor
    past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None


class Qwen3LM(nn.Module):
    def __init__(self, config: LLMConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size) # 一张 (vocab_size, hidden_size) 的 embedding 表
        self.layers = nn.ModuleList(
            Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers) # default num_hidden_layers = 2
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None, # shape is (batch_size, past_length + time_steps) padding mask, 1 for tokens that are NOT masked, 0 for tokens that are masked
        position_ids: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None, # length is num_hidden_layers, each element is a tuple of (past_k, past_v) where past_k.shape is (batch_size, num_kv_head, past_length, head_dim) and past_v.shape is (batch_size, num_kv_head, past_length, head_dim)
        use_cache: bool = False,
    ) -> QwenLMOutput:
        # inputs_embeds.shape is (batch_size, max_length=max_across_batch(prompt_length+speech_length+target_length), hidden_size)
        # attention_mask.shape is (batch_size, max_length=max_across_batch(prompt_length+speech_length+target_length))
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("provide exactly one of input_ids or inputs_embeds")
        x = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        batch_size, time_steps, _ = x.shape
        past_length = 0 if past_key_values is None else past_key_values[0][0].shape[-2]
        
        cache_positions = torch.arange(past_length, past_length + time_steps, device=x.device) 
        # 当前这次输入的 token，分别处于完整序列的哪些位置
        # 形状为 （time_steps = max_length = max_across_batch(prompt_length+speech_length+target_length),)
        
        if position_ids is None:
            positions = cache_positions # (time_steps,)
        else:
            if position_ids.shape != (batch_size, time_steps):
                raise ValueError("position_ids must have shape [B, T]")
            positions = position_ids # (batch_size, time_steps)

        cos, sin = rotary_embeddings(positions, self.config.head_dim, self.config.rope_theta, x.dtype)
        # cos.shape is (1, 1, time_steps, head_dim) or (batch_size, 1, time_steps, head_dim)
        # sin.shape is (1, 1, time_steps, head_dim) or (batch_size, 1, time_steps, head_dim)

        key_positions = torch.arange(past_length + time_steps, device=x.device) # (past_length + time_steps,)

        causal = key_positions[None, :] <= cache_positions[:, None] # (time_steps, past_length + time_steps) 当前 query 不能看未来 key

        if attention_mask is None:
            attention_mask = torch.ones(
                batch_size, past_length + time_steps, dtype=torch.bool, device=x.device
            ) # (batch_size, past_length + time_steps)
    
        allowed = causal[None, None] & attention_mask[:, None, None, :].bool()
        # causal[None, None].shape is (1, 1, time_steps, past_length + time_steps)
        # attention_mask[:, None, None, :].bool().shape is (batch_size, 1, 1, past_length + time_steps)
        # allowed.shape is (batch_size, 1, time_steps, past_length + time_steps) 其中 dim 1 表示 attention head（1 表示广播到所有 head）

        presents = [] if use_cache else None
        for index, layer in enumerate(self.layers):
            past = None if past_key_values is None else past_key_values[index]
            x, present = layer(x, cos, sin, allowed, past)
            if presents is not None:
                presents.append(present)
        return QwenLMOutput(self.lm_head(self.norm(x)), presents)
