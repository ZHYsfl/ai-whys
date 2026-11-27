# Copyright (c) Kyutai, all rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Derived from kyutai-labs/moshi; see NOTICE and LICENSES/.

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch
from torch import nn

from .config import MoshiConfig
from .streaming import StreamingContainer
from .transformer import StreamingTransformer, create_norm_fn


@dataclass
class LMOutput:
    logits: torch.Tensor
    mask: torch.Tensor
    text_logits: torch.Tensor
    text_mask: torch.Tensor


def _delay_sequence(
    delays: tuple[int, ...], tensor: torch.Tensor, padding: torch.Tensor
) -> torch.Tensor:
    # delays default (0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1) 17个数
    # tensor.shape is (B,17,T), padding.shape is (B,17,1) _get_initial_token造的
    # Each stream has its own delay; padding supplies initial tokens at negative times.
    outputs = []
    for stream, delay in enumerate(delays):
        # stream = 1,2,...,16, delay = 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1
        line = tensor[:, stream].roll(delay, dims=1)
        if delay:
            line[:, :delay] = padding[:, stream]
        outputs.append(line)
    return torch.stack(outputs, dim=1) # (B,17,T) 17个流，每个流的时间轴都被roll了不同的delay，前面delay个时间步被padding替换


def _undelay_sequence(
    delays: tuple[int, ...], tensor: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # delays default (0, 1, 1, 1, 1, 1, 1, 1)
    # tensor.shape default (batch_size, 8, new_num_frames-1, 64)
    batch_size, streams, time_steps = tensor.shape[:3]
    mask = torch.ones(
        batch_size, streams, time_steps, device=tensor.device, dtype=torch.bool
    )
    outputs = []
    for stream, delay in enumerate(delays):
        line = tensor[:, stream].roll(-delay, dims=1) # line.shape is (batch_size, new_num_frames-1, 64)
        if delay:
            line[:, -delay:] = float("nan")
            mask[:, stream, -delay:] = False
        outputs.append(line)
    return torch.stack(outputs, dim=1), mask


class ScaledEmbedding(nn.Embedding):
    """Embedding with a zero-vector sentinel for invalid/padded positions."""

    def __init__(self, num_embeddings: int, embedding_dim: int, zero_idx: int = -1) -> None:
        # num_embedings default 65, embedding_dim default 64, zero_idx default -1
        super().__init__(num_embeddings, embedding_dim)
        self.zero_idx = zero_idx

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # input.shape is (batch_size, time_steps)
        is_zero = input == self.zero_idx # is_zero.shape is (batch_size, time_steps)
        output = super().forward(input.clamp_min(0)) # output.shape is (batch_size, time_steps, embedding_dim)
        # 是零向量，而不是一个“pad向量”
        return torch.where(is_zero[..., None], torch.zeros_like(output), output) # is_zero[..., None] broadcast to (batch_size, time_steps, embedding_dim), output.shape is (batch_size, time_steps, embedding_dim)


class MoshiLM(StreamingContainer):
    """The complete multi-stream RQ-Transformer used by Moshi."""

    def __init__(self, config: MoshiConfig | None = None) -> None:
        super().__init__()
        self.config = config or MoshiConfig()
        self.n_q = self.config.n_q # default 16
        self.dep_q = self.config.dep_q # default 8
        self.card = self.config.card # default 64
        self.text_card = self.config.text_card # default 260
        self.delays = self.config.delays # default (0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1)
        self.dim = self.config.dim # default 64
        self.existing_text_padding_id = self.config.existing_text_padding_id # default 3
        self.existing_text_end_padding_id = self.config.existing_text_end_padding_id # default 0

        self.emb = nn.ModuleList(
            [ScaledEmbedding(self.card + 1, self.dim) for _ in range(self.n_q)]
        ) # 16个 (65, 64) embedding
        self.text_emb = ScaledEmbedding(self.text_card + 1, self.dim) # (261, 64) embedding

        self.text_linear = nn.Linear(self.dim, self.text_card, bias=False)
        self.transformer = StreamingTransformer(
            d_model=self.dim, # default 64
            num_heads=self.config.num_heads, # default 4
            num_layers=self.config.num_layers, # default 2
            dim_feedforward=int(self.config.hidden_scale * self.dim), # default 4*64=256
            causal=True,
            context=self.config.context, # default 3000
            positional_embedding="rope",
            max_period=self.config.max_period, # default 10_000.0
            norm="rms_norm_f32",
            gating="silu",
        )
        self.out_norm = create_norm_fn("rms_norm_f32", self.dim)

        input_projections = self.dep_q if self.config.depformer_multi_linear else 1 # default 8
        if self.config.depformer_weights_per_step_schedule is not None:
            input_projections = max(self.config.depformer_weights_per_step_schedule) + 1
        self.depformer_in = nn.ModuleList(
            [
                nn.Linear(self.dim, self.config.depformer_dim, bias=False)
                for _ in range(input_projections)
            ]
        )
        self.depformer_emb = nn.ModuleList(
            [
                ScaledEmbedding(self.card + 1, self.config.depformer_dim)
                for _ in range(self.dep_q - 1)
            ]
        )
        self.depformer_text_emb = ScaledEmbedding(
            self.text_card + 1, self.config.depformer_dim
        ) # (261, 32) embedding
        self.depformer_norms = nn.ModuleList(
            [
                nn.Identity()
                if self.config.depformer_norm is None
                else create_norm_fn(self.config.depformer_norm, self.config.depformer_dim)
                for _ in range(self.dep_q)
            ]
        )
        self.depformer = StreamingTransformer(
            d_model=self.config.depformer_dim, # default 32
            num_heads=self.config.depformer_num_heads, # default 4
            num_layers=self.config.depformer_num_layers, # default 2
            dim_feedforward=self.config.depformer_dim_feedforward, # default 128
            causal=True,
            context=self.config.depformer_context, # default 8（包括自己）
            positional_embedding="none",
            norm="rms_norm_f32",
            gating="silu",
            weights_per_step=self.dep_q if self.config.depformer_weights_per_step else 0, # default 8
            weights_per_step_schedule=self.config.depformer_weights_per_step_schedule,
        )
        self.depformer.set_streaming_detached(True)
        self.linears = nn.ModuleList(
            [
                nn.Linear(self.config.depformer_dim, self.card, bias=False)
                for _ in range(self.dep_q)
            ]
        )

    @property
    def  num_codebooks(self) -> int:
        return self.n_q + 1

    @property
    def num_audio_codebooks(self) -> int:
        return self.n_q

    @property
    def audio_offset(self) -> int:
        return 1

    @property
    def initial_token_id(self) -> int:
        return self.card

    @property
    def text_initial_token_id(self) -> int:
        return self.text_card

    @property
    def zero_token_id(self) -> int:
        return -1

    @property
    def ungenerated_token_id(self) -> int:
        return -2

    @property
    def text_padding_token_id(self) -> int:
        return self.existing_text_padding_id

    @property
    def end_of_text_padding_id(self) -> int:
        return self.existing_text_end_padding_id

    def _get_initial_token(self) -> torch.Tensor:
        parameter = self.text_emb.weight
        text = torch.full(
            (1, 1, 1), self.text_initial_token_id, device=parameter.device, dtype=torch.long
        )
        audio = torch.full(
            (1, self.n_q, 1), self.initial_token_id, device=parameter.device, dtype=torch.long
        )
        return torch.cat((text, audio), dim=1) # (1, 1+n_q, 1) = (1, 17, 1) 
        # [ [ [260], [64], [64], [64], [64], [64], [64], [64], [64], [64], [64], [64], [64], [64], [64], [64], [64] ] ]

    def forward(self, codes: torch.Tensor) -> LMOutput:
        """Teacher-forced training forward; codes are [B, 1+16, T]."""
        if codes.ndim != 3 or codes.shape[1] != self.num_codebooks:
            raise ValueError(
                f"expected codes [B, {self.num_codebooks}, T], got {tuple(codes.shape)}"
            )
        initial = self._get_initial_token().expand(codes.shape[0], -1, -1) # (B, 17, 1)
        delayed = _delay_sequence(self.delays, codes, initial)
        delayed = torch.cat((initial, delayed), dim=2) # （B, 17, T)
        temporal, text_logits = self.forward_text(delayed[:, :, :-1])
        logits = self.forward_depformer_training(delayed[:, :, 1:], temporal)
        logits, mask = _undelay_sequence(self.delays[1 : self.dep_q + 1], logits)
        mask &= codes[:, 1 : self.dep_q + 1] != self.zero_token_id
        text_logits, text_mask = _undelay_sequence(self.delays[:1], text_logits)
        text_mask &= codes[:, :1] != self.zero_token_id
        return LMOutput(logits, mask, text_logits, text_mask)

    def forward_text(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # sequence.shape is (B, 17, T-1)
        if sequence.ndim != 3 or sequence.shape[1] != self.num_codebooks:
            raise ValueError("invalid multi-stream input")
        embeddings = [self.text_emb(sequence[:, 0])] # embeddings.shape is (B,T-1,D)
        embeddings.extend(
            self.emb[index](sequence[:, index + self.audio_offset])
            for index in range(self.num_audio_codebooks)
        ) # 17 个 (B,T-1,D)
        input_ = torch.stack(embeddings, dim=0).sum(dim=0) # (B,T-1,D) sum over 17 streams
        temporal = self.out_norm(self.transformer(input_))
        return temporal, self.text_linear(temporal)[:, None]

    def _depformer_projection(self, codebook_index: int, temporal: torch.Tensor) -> torch.Tensor:
        # codebook_index default 0-7, temporal.shape is (batch_size, new_num_frames-1, dimension=64)
        projection_index = codebook_index if len(self.depformer_in) > 1 else 0
        if self.config.depformer_weights_per_step_schedule is not None:
            projection_index = self.config.depformer_weights_per_step_schedule[codebook_index]
        return self.depformer_in[projection_index](temporal)

    def forward_depformer_training(
        self, sequence: torch.Tensor, temporal: torch.Tensor
    ) -> torch.Tensor:
        # sequence.shape is (batch_size, 17=1+8+8, new_num_frames-1)
        # temporal.shape is (batch_size, new_num_frames-1, dimension=64)
        batch_size, _, time_steps = sequence.shape
        inputs = []
        for codebook_index in range(self.dep_q): # 0-7
            if codebook_index == 0:
                token = self.depformer_text_emb(sequence[:, 0])
            else:
                token = self.depformer_emb[codebook_index - 1](
                    sequence[:, codebook_index]
                )
            inputs.append(token + self._depformer_projection(codebook_index, temporal))
        # inputs 是 8 个 (batch_size, new_num_frames-1, depformer_dimension=32)
        # [B,T,K,D] -> [B*T,K,D]: time is batched; depth remains autoregressive.
        depth_input = torch.stack(inputs, dim=2).view(
            batch_size * time_steps, self.dep_q, -1
        ) # (batch_size, new_num_frames-1, 8, depformer_dimension=32) -> (batch_size * (new_num_frames-1), 8, depformer_dimension=32)
        depth_output = self.depformer(depth_input)
        logits = [
            self.linears[index](self.depformer_norms[index](depth_output[:, index])).view(
                batch_size, time_steps, self.card
            )
            for index in range(self.dep_q)
        ]
        return torch.stack(logits, dim=1)

    def forward_depformer(
        self, codebook_index: int, sequence: torch.Tensor, temporal: torch.Tensor
    ) -> torch.Tensor:
        # sequence.shape is (batch_size, 1, 1), temporal.shape is (batch_size, new_num_frames-1, dimension=64)
        if sequence.shape[1:] != (1, 1) or temporal.shape[1] != 1:
            raise ValueError("streaming Depformer consumes exactly one token at a time")
        if codebook_index == 0:
            token = self.depformer_text_emb(sequence[:, 0])
        else:
            token = self.depformer_emb[codebook_index - 1](sequence[:, 0])
        depth_input = token + self._depformer_projection(codebook_index, temporal)
        depth_output = self.depformer(depth_input)
        logits = self.linears[codebook_index](
            self.depformer_norms[codebook_index](depth_output)
        ) # (batch_size, 1, card=64)
        return logits[:, None] # (batch_size, 1, 1, card=64)


def sample_token(
    logits: torch.Tensor,
    use_sampling: bool = True,
    temperature: float = 0.8,
    top_k: int = 25,
) -> torch.Tensor:
    # Input [B,1,1,vocab] or [B,vocab]; output retains [B,1,1] for 4D input.
    squeeze = logits.ndim == 4
    logits = logits.float().reshape(logits.shape[0], -1)
    if not use_sampling or temperature <= 0:
        tokens = logits.argmax(dim=-1) # 贪心
    else:
        logits = logits / temperature
        if top_k > 0 and top_k < logits.shape[-1]:
            threshold = logits.topk(top_k, dim=-1).values[:, -1, None] # （B,1) 第k名
            logits = logits.masked_fill(logits < threshold, -float("inf"))
        tokens = torch.multinomial(logits.softmax(dim=-1), 1)[:, 0] # (B,)
    return tokens[:, None, None] if squeeze else tokens # (B,1,1) or (B,)


@dataclass
class GeneratorState:
    batch_size: int
    cache: torch.Tensor
    initial: torch.Tensor
    offset: int = 0


class MoshiGenerator:
    """One real-time step: user tokens in, Moshi text + 8 audio tokens out."""

    def __init__(
        self,
        model: MoshiLM,
        use_sampling: bool = True,
        temperature: float = 0.8,
        text_temperature: float = 0.7,
        top_k: int = 250,
        text_top_k: int = 25,
    ) -> None:
        self.lm_model = model
        self.use_sampling = use_sampling
        self.temperature = temperature
        self.text_temperature = text_temperature
        self.top_k = top_k
        self.text_top_k = text_top_k
        self._state: GeneratorState | None = None

    @contextmanager
    def streaming(self, batch_size: int) -> Iterator["MoshiGenerator"]:
        if self._state is not None:
            raise RuntimeError("generator streaming is already active")
        model = self.lm_model
        cache = torch.full(
            (batch_size, model.num_codebooks, model.config.max_delay + 2),
            model.ungenerated_token_id,
            device=model.text_emb.weight.device,
            dtype=torch.long,
        )
        initial = model._get_initial_token().expand(batch_size, -1, -1)
        with model.streaming(batch_size):
            self._state = GeneratorState(batch_size, cache, initial)
            try:
                yield self
            finally:
                self._state = None

    @torch.no_grad()
    def step(self, user_tokens: torch.Tensor) -> torch.Tensor | None:
        # user_tokens.shape is (batch_size, 8, 1)
        state = self._state
        if state is None:
            raise RuntimeError("wrap step calls in 'with generator.streaming(batch_size):'")
        model = self.lm_model
        expected = (state.batch_size, model.config.user_codebooks, 1)
        if tuple(user_tokens.shape) != expected:
            raise ValueError(f"expected user tokens {expected}, got {tuple(user_tokens.shape)}")
        cache = state.cache # (batch_size, 17, 3)
        delays = torch.tensor(model.delays, device=cache.device)
        user_delays = delays[model.config.user_audio_offset :] # (8,)
        write_positions = (state.offset + user_delays) % cache.shape[-1] # (8,)
        user_cache = cache[:, model.config.user_audio_offset :] # (batch_size, 8, 3)
        user_cache.scatter_(
            2,
            write_positions[None, :, None].expand_as(user_tokens), # (batch_size, 8, 1)
            user_tokens, # (batch_size, 8, 1)
        )

        is_initial = state.offset <= delays # (17,)
        read_position = torch.full(
            (state.batch_size, model.num_codebooks, 1),
            state.offset % cache.shape[-1],
            device=cache.device,
            dtype=torch.long,
        )
        input_ = cache.gather(2, read_position) # (batch_size, 17, 1)
        input_ = torch.where(is_initial[None, :, None], state.initial, input_) # (batch_size, 17, 1)
        temporal, text_logits = model.forward_text(input_)
        text_token = sample_token(
            text_logits,
            self.use_sampling,
            self.text_temperature,
            self.text_top_k,
        )[:, 0, 0] # (batch_size,)

        audio_tokens = []
        previous = text_token # (batch_size,)
        with model.depformer.streaming(state.batch_size):
            for codebook_index in range(model.dep_q):
                logits = model.forward_depformer(
                    codebook_index, previous[:, None, None], temporal
                ) # (batch_size, 1, 1, card=64)
                next_token = sample_token(
                    logits,
                    self.use_sampling,
                    self.temperature,
                    self.top_k,
                )[:, 0, 0] # (batch_size,)
                audio_tokens.append(next_token)
                previous = next_token
        audio = torch.stack(audio_tokens, dim=1) # (batch_size, 8)

        state.offset += 1
        write_position = state.offset % cache.shape[-1]
        cache[:, :1, write_position] = text_token[:, None] # (batch_size, 1)
        cache[:, 1 : model.dep_q + 1, write_position] = audio # (batch_size, 8)
        if state.offset <= model.config.max_delay: # 只有第 0 step
            return None

        generated_delays = delays[: model.dep_q + 1]
        output_positions = (
            state.offset - model.config.max_delay + generated_delays
        ) % cache.shape[-1]
        return cache.gather(
            2,
            output_positions[None, :, None].expand(state.batch_size, -1, -1),
        ) # (batch_size, 9, 1) = (batch_size, 1+8, 1)
