from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .features import LogMelSpectrogram, lengths_to_mask
from .modern_config import AudioConfig, LLMConfig


def sinusoidal_positions(time_steps: int, dimension: int, device: torch.device) -> torch.Tensor:
    positions = torch.arange(time_steps, device=device).float()[:, None] # (time_steps, 1)
    frequencies = torch.exp(
        torch.arange(0, dimension, 2, device=device).float()
        * (-torch.log(torch.tensor(10_000.0, device=device)) / dimension)
    ) # frequencies[i] = 10000^(-2i/dimension) i = 0,1,...,dimension/2-1
    encoding = torch.zeros(time_steps, dimension, device=device)
    encoding[:, 0::2] = torch.sin(positions * frequencies)
    encoding[:, 1::2] = torch.cos(positions * frequencies)
    return encoding


class AudioSelfAttention(nn.Module):
    def __init__(self, config: AudioConfig) -> None:
        super().__init__()
        self.num_heads = config.encoder_heads # default 4
        self.head_dim = config.d_model // config.encoder_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.output = nn.Linear(config.d_model, config.d_model)

    def forward(self, x: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
        # x.shape is (batch_size, time_steps, d_model)
        # allowed.shape is (batch_size, time_steps, time_steps)
        batch_size, time_steps, dimension = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch_size, time_steps, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, time_steps, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, time_steps, self.num_heads, self.head_dim).transpose(1, 2)
        # q,k,v shape is (batch_size, num_head, time_steps,d_model/num_head)
        attended = F.scaled_dot_product_attention(
            q, k, v, attn_mask=allowed[:, None], scale=self.head_dim**-0.5
        )
        attended = attended.transpose(1, 2).contiguous().view(batch_size, time_steps, dimension)
        return self.output(attended)


class AudioEncoderLayer(nn.Module):
    def __init__(self, config: AudioConfig) -> None:
        super().__init__()
        self.self_attention_norm = nn.LayerNorm(config.d_model)
        self.self_attention = AudioSelfAttention(config)
        self.ffn_norm = nn.LayerNorm(config.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.ffn_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ffn_dim, config.d_model),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
        # x.shape is (batch_size, time_steps, d_model)
        # allowed.shape is (batch_size, time_steps, time_steps)
        x = x + self.dropout(self.self_attention(self.self_attention_norm(x), allowed))
        return x + self.dropout(self.ffn(self.ffn_norm(x)))


class Qwen3AudioEncoder(nn.Module):
    def __init__(self, config: AudioConfig) -> None:
        super().__init__()
        self.config = config
        self.frontend = LogMelSpectrogram(config.feature_config())
        channels = config.downsample_channels # default 32
        self.conv1 = nn.Conv2d(1, channels, 3, 2, padding=1) # (t+2*1-3)//2+1 = (t+1)//2
        self.conv2 = nn.Conv2d(channels, channels, 3, 2, padding=1)
        self.conv3 = nn.Conv2d(channels, channels, 3, 2, padding=1)
        self.conv_out = nn.Linear(channels * config.downsampled_mels, config.d_model, bias=False)
        self.layers = nn.ModuleList(
            AudioEncoderLayer(config) for _ in range(config.encoder_layers)
        )
        self.ln_post = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        waveforms: torch.Tensor, # (batch_size, time_steps=sample_rate*seconds)
        waveform_lengths: torch.Tensor,
        attention_window: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        window = self.config.attention_window if attention_window is None else attention_window # default 25
        if window < 0:
            raise ValueError("attention_window must be >= 0; zero means full context")
        features, frame_lengths = self.frontend(waveforms, waveform_lengths)
        output_lengths = frame_lengths # (batch_size,)
        x = features[:, None] # (batch_size, 1, num_window, num_mel)
        for conv in (self.conv1, self.conv2, self.conv3):
            x = F.gelu(conv(x))
            output_lengths = (output_lengths + 1) // 2
            mask = lengths_to_mask(output_lengths, x.shape[2])
            x = x.masked_fill(~mask[:, None, :, None], 0.0)
        if (output_lengths < 1).any():
            raise ValueError("an utterance is too short for 8x AuT subsampling")
        batch_size, channels, time_steps, mel_steps = x.shape
        x = self.conv_out(
            x.permute(0, 2, 1, 3).contiguous().view(
                batch_size, time_steps, channels * mel_steps
            ) # [B,C,T,F] -> [B,T,C*F]；第二维才是时间，不能把 Mel 当时间。
        ) # (batch_size, time_steps, d_model)
        x = self.dropout(x + sinusoidal_positions(time_steps, self.config.d_model, x.device)) # (batch_size, time_steps, d_model)
        if window:
            positions = torch.arange(time_steps, device=x.device)
            chunks = positions // window # 每个位置的块编号
            allowed = mask[:, None, :] & (chunks[:, None] == chunks[None, :])[None] # (batch_size, 1, time_steps) & (1, time_steps, time_steps) -> (batch_size, time_steps, time_steps)
        else:
            allowed = mask[:, None, :].expand(batch_size, time_steps, time_steps) # 此时不划分时间块，只屏蔽 padding (batch_size, 1, time_steps) -> (batch_size, time_steps, time_steps) 
        for layer in self.layers:
            x = layer(x, allowed)
        x = self.ln_post(x).masked_fill(~mask[:, :, None], 0.0)
        return x, output_lengths, mask


@dataclass
class AuTOutput:
    loss: torch.Tensor # 0-dim tensor
    logits: torch.Tensor # (batch_size, max_length-1, vocab_size)
    memory_lengths: torch.Tensor # (batch_size,)


class AuT(nn.Module):
    def __init__(self, audio_config: AudioConfig, llm_config: LLMConfig) -> None:
        super().__init__()
        if audio_config.d_model != llm_config.hidden_size:
            raise ValueError("AuT audio and decoder hidden sizes must match")
        self.audio_config = audio_config
        self.llm_config = llm_config
        self.audio_encoder = Qwen3AudioEncoder(audio_config)
        self.token_embedding = nn.Embedding(llm_config.vocab_size, llm_config.hidden_size)
        layer = nn.TransformerDecoderLayer(
            llm_config.hidden_size,
            llm_config.num_attention_heads,
            llm_config.intermediate_size,
            dropout=audio_config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, llm_config.num_hidden_layers)
        self.output = nn.Linear(llm_config.hidden_size, llm_config.vocab_size, bias=False)

    def forward(
        self,
        waveforms: torch.Tensor, # (batch_size, time_steps=sample_rate*seconds)
        waveform_lengths: torch.Tensor, # (batch_size,)
        target_ids: torch.Tensor, # # (batch_size, max_length)
        target_lengths: torch.Tensor, # (batch_size,)
    ) -> AuTOutput:
        memory, memory_lengths, memory_mask = self.audio_encoder(waveforms, waveform_lengths)
        decoder_input = target_ids[:, :-1] # (batch_size, max_length-1)
        labels = target_ids[:, 1:] # (batch_size, max_length-1)
        embedded = self.token_embedding(decoder_input) # (batch_size, max_length-1, hidden_size)
        embedded = embedded + sinusoidal_positions(
            embedded.shape[1], embedded.shape[2], embedded.device
        )[None] # (batch_size, max_length-1, hidden_size)
        target_mask = lengths_to_mask(target_lengths - 1, decoder_input.shape[1]) # (batch_size, max_length-1)
        causal = torch.triu(
            torch.ones(
                decoder_input.shape[1],
                decoder_input.shape[1],
                device=decoder_input.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        ) # causal.shape is (max_length-1, max_length-1)
        logits = self.decoder(
            embedded, # (batch_size, max_length-1, hidden_size)
            memory, # (batch_size, time_steps, hidden_size)
            tgt_mask=causal, # 文本 self attention , 屏蔽未来文本
            tgt_key_padding_mask=~target_mask, # 文本 self attention , 屏蔽文本 padding
            memory_key_padding_mask=~memory_mask, # cross attention , 屏蔽音频 padding
        ) # logits.shape is (batch_size, max_length-1, hidden_size)
        logits = self.output(logits) # logits.shape is (batch_size, max_length-1, vocab_size)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), # (batch_size*(max_length-1), vocab_size)
            labels.reshape(-1), # (batch_size*(max_length-1),)
            ignore_index=0 # 当前 token ID 0 = <PAD> , ignore_index=0 排除 padding
        ) # loss is 0-dim tensor
        return AuTOutput(loss, logits, memory_lengths)
