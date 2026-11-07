from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .aut import Qwen3AudioEncoder
from .llm_text import LanguageTokenizer
from .modern_config import ModernASRConfig
from .qwen3_lm import Qwen3LM


@dataclass
class SpeechLLMOutput:
    loss: torch.Tensor
    logits: torch.Tensor
    speech_lengths: torch.Tensor


class Qwen3ASR(nn.Module):
    def __init__(self, config: ModernASRConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModernASRConfig()
        self.tokenizer = LanguageTokenizer()
        if self.tokenizer.vocab_size != self.config.llm.vocab_size:
            raise ValueError("LLM vocab_size does not match LanguageTokenizer")
        self.audio_encoder = Qwen3AudioEncoder(self.config.audio)
        self.projector = nn.Linear(self.config.audio.d_model, self.config.llm.hidden_size, bias=False)
        self.llm = Qwen3LM(self.config.llm)

    def speech_embeddings(
        self, waveforms: torch.Tensor, waveform_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded, output_lengths, mask = self.audio_encoder(waveforms, waveform_lengths)
        embeddings = self.projector(encoded).masked_fill(~mask[:, :, None], 0.0)
        return embeddings, output_lengths

    def _build_inputs(
        self,
        speech_embeddings: torch.Tensor,
        speech_lengths: torch.Tensor,
        transcripts: tuple[str, ...] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if speech_lengths.shape != (speech_embeddings.shape[0],):
            raise ValueError("speech_lengths must have shape [B]")
        if (speech_lengths <= 0).any() or (speech_lengths > speech_embeddings.shape[1]).any():
            raise ValueError("speech lengths must be in [1, T]")
        # speech_embeddings.shape is (batch_size, time_steps, hidden_size)
        # speech_lengths.shape is (batch_size,)
        # the length of transcripts must match the batch size
        prompt = self.llm.embed_tokens(
            torch.tensor(self.tokenizer.prompt_ids, device=speech_embeddings.device)
        ) # prompt.shape is (prompt_length = 3, hidden_size)
        sequences = []
        labels = []
        for index, speech in enumerate(speech_embeddings):
            speech = speech[: int(speech_lengths[index])]
            parts = [prompt, speech]
            if transcripts is not None:
                target_ids = self.tokenizer.encode(transcripts[index]) + [self.tokenizer.eos_id]
                target = self.llm.embed_tokens(
                    torch.tensor(target_ids, device=speech_embeddings.device)
                ) # target.shape is (target_length, hidden_size)
                parts.append(target) # [prompt, speech, target]
                masked_length = prompt.shape[0] + speech.shape[0] # prompt_length + speech_length
                labels.append(
                    torch.cat(
                        (
                            torch.full((masked_length,), -100, device=target.device),
                            torch.tensor(target_ids, device=target.device),
                        )
                    ) # (prompt_length + speech_length + target_length,)
                )
            sequences.append(
                torch.cat(parts, dim=0) # (prompt_length + speech_length + target_length, hidden_size)
            )

        # now the length of sequences is batch_size, and each sequence has shape (prompt_length + speech_length + target_length, hidden_size)
        # now the length of labels is batch_size, and each label has shape (prompt_length + speech_length + target_length,)
        max_length = max(sequence.shape[0] for sequence in sequences)
        # max_length is the maximum length of prompt_length + speech_length + target_length across the batch
        padded = []
        attention_mask = []
        for sequence in sequences:
            padding = max_length - sequence.shape[0]
            padded.append(F.pad(sequence, (0, 0, 0, padding))) # F.pad() 对多维输入的 padding 参数，规则是从最后一维开始，每一维依次写“左边补多少、右边补多少”
            attention_mask.append(
                torch.cat(
                    (
                        torch.ones(sequence.shape[0], dtype=torch.bool, device=sequence.device),
                        torch.zeros(padding, dtype=torch.bool, device=sequence.device),
                    )
                ) # (max_length,)
            )
        label_tensor = None
        if labels:
            label_tensor = torch.stack(
                [F.pad(label, (0, max_length - label.shape[0]), value=-100) for label in labels]
            ).long()
        
        return torch.stack(padded), torch.stack(attention_mask), label_tensor
        # torch.stack(padded).shape is (batch_size, max_length, hidden_size)
        # torch.stack(attention_mask).shape is (batch_size, max_length)
        # label_tensor.shape is (batch_size, max_length)

    def forward(
        self,
        waveforms: torch.Tensor,
        waveform_lengths: torch.Tensor,
        transcripts: tuple[str, ...],
    ) -> SpeechLLMOutput:
        speech, speech_lengths = self.speech_embeddings(waveforms, waveform_lengths)
        # speech.shape is (batch_size, time_steps, hidden_size)
        # speech_lengths.shape is (batch_size,)
        inputs, attention_mask, labels = self._build_inputs(speech, speech_lengths, transcripts)
        # inputs.shape is (batch_size, max_length, hidden_size)
        # attention_mask.shape is (batch_size, max_length)
        # labels.shape is (batch_size, max_length)
        
        output = self.llm(inputs_embeds=inputs, attention_mask=attention_mask)
        # next-token：位置 i 的 logits 预测位置 i+1 的 label；prompt/speech 的 label 是 -100。
        shift_logits = output.logits[:, :-1].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]), # (batch_size * (max_length - 1), vocab_size)
            shift_labels.view(-1), # （batch_size * (max_length - 1), ）
            ignore_index=-100, # 忽略 prompt、音频和 padding，只对文本字符与 EOS 计算损失。
        )
        return SpeechLLMOutput(loss, output.logits, speech_lengths)

    @torch.no_grad()
    def transcribe(
        self,
        waveforms: torch.Tensor,
        waveform_lengths: torch.Tensor,
        max_new_tokens: int = 128,
    ) -> list[str]:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        speech, speech_lengths = self.speech_embeddings(waveforms, waveform_lengths)
        inputs, attention_mask, _ = self._build_inputs(speech, speech_lengths, None)
        output = self.llm(
            inputs_embeds=inputs, attention_mask=attention_mask, use_cache=True
        )
        past = output.past_key_values
        last_indices = attention_mask.sum(dim=1) - 1
        next_token = output.logits[
            torch.arange(inputs.shape[0], device=inputs.device), last_indices
        ].argmax(dim=-1)
        generated = []
        finished = torch.zeros_like(next_token, dtype=torch.bool)
        for _ in range(max_new_tokens):
            generated.append(next_token)
            finished |= next_token == self.tokenizer.eos_id
            if torch.all(finished):
                break
            attention_mask = torch.cat(
                (
                    attention_mask,
                    torch.ones(
                        attention_mask.shape[0], 1, dtype=torch.bool, device=attention_mask.device
                    ),
                ),
                dim=1,
            )
            output = self.llm(
                input_ids=next_token[:, None],
                attention_mask=attention_mask,
                position_ids=(attention_mask.sum(dim=1) - 1)[:, None],
                past_key_values=past,
                use_cache=True,
            )
            past = output.past_key_values
            next_token = output.logits[:, -1].argmax(dim=-1)
            next_token = next_token.masked_fill(finished, self.tokenizer.eos_id)
        return [
            self.tokenizer.normalize(self.tokenizer.decode(sequence))
            for sequence in torch.stack(generated, dim=1).cpu()
        ]
