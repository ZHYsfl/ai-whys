# Derived from kyutai-labs/moshi (MIT); see NOTICE and LICENSES/MIT-moshi.txt.

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .mimi import MimiModel
from .model import MoshiGenerator, MoshiLM
from .text import TextTokenizer


@dataclass
class DialogueOutput:
    waveform: torch.Tensor
    audio_tokens: torch.Tensor
    text_tokens: list[list[int]]
    text: list[str]


class MoshiDialogue:
    """Full-duplex data flow without hiding any of the three model stages."""

    def __init__(
        self,
        mimi: MimiModel,
        model: MoshiLM,
        tokenizer: TextTokenizer,
        use_sampling: bool = True,
        temperature: float = 0.8,
        text_temperature: float = 0.7,
    ) -> None:
        if mimi.cardinality != model.card:
            raise ValueError("Mimi and Moshi audio vocabularies differ")
        if mimi.frame_rate <= 0:
            raise ValueError("invalid Mimi frame rate")
        mimi.set_num_codebooks(model.dep_q)
        self.mimi = mimi
        self.model = model
        self.tokenizer = tokenizer
        self.generator = MoshiGenerator(
            model,
            use_sampling=use_sampling,
            temperature=temperature,
            text_temperature=text_temperature,
        )

    @torch.no_grad()
    def respond(self, user_waveform: torch.Tensor) -> DialogueOutput:
        # user_waveform.shape is (batch_size,input_channels=1, time_steps = training : sample_rate*seconds   inferring : frame_size)
        if user_waveform.ndim != 3 or user_waveform.shape[1] != 1:
            raise ValueError(f"expected user waveform [B,1,T], got {tuple(user_waveform.shape)}")
        batch_size = user_waveform.shape[0]
        padding = (-user_waveform.shape[-1]) % self.mimi.frame_size
        user_waveform = F.pad(user_waveform, (0, padding))
        self.mimi.freeze()
        self.model.eval()

        waveform_chunks = []
        audio_chunks = []
        text_tokens = [[] for _ in range(batch_size)]
        with self.mimi.streaming(batch_size), self.generator.streaming(batch_size):
            for frame in user_waveform.split(self.mimi.frame_size, dim=-1):
                user_codes = self.mimi.encode(frame)
                tokens = self.generator.step(user_codes) # (b,9,1)
                if tokens is None:
                    continue
                audio_codes = tokens[:, 1:] # (b,8,1)
                audio_chunks.append(audio_codes)
                waveform_chunks.append(self.mimi.decode(audio_codes))
                for batch_index, token in enumerate(tokens[:, 0, 0].tolist()):
                    if token not in {
                        self.model.text_padding_token_id,
                        self.model.end_of_text_padding_id,
                    }:
                        text_tokens[batch_index].append(token)

        if not waveform_chunks:
            waveform = torch.empty(
                batch_size, 1, 0, device=user_waveform.device, dtype=user_waveform.dtype
            )
            audio_tokens = torch.empty(
                batch_size,
                self.model.dep_q,
                0,
                device=user_waveform.device,
                dtype=torch.long,
            )
        else:
            waveform = torch.cat(waveform_chunks, dim=-1) # (batch_size, 1, time_steps)
            audio_tokens = torch.cat(audio_chunks, dim=-1) # (batch_size, 8, time_steps/frame_size)
        return DialogueOutput(
            waveform, # (batch_size, 1, time_steps)
            audio_tokens, # (batch_size, 8, time_steps/frame_size)
            text_tokens, # batch_size 个列表，每个是一个句子的文本 token id 列表
            [self.tokenizer.decode(tokens) for tokens in text_tokens], # batch_size 个字符串，每个是一个句子的文本
        )
