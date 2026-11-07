from dataclasses import dataclass

import torch
from torch import nn

from .config import ASRConfig
from .conformer import ConformerEncoder
from .features import LogMelSpectrogram, spec_augment
from .text import CharTokenizer


@dataclass
class ASROutput:
    logits: torch.Tensor
    output_lengths: torch.Tensor


class ConformerCTC(nn.Module):
    def __init__(
        self,
        config: ASRConfig | None = None,
        tokenizer: CharTokenizer | None = None,
    ) -> None:
        super().__init__()
        self.config = config or ASRConfig()
        self.tokenizer = tokenizer or CharTokenizer()
        self.frontend = LogMelSpectrogram(self.config)
        self.encoder = ConformerEncoder(self.config)
        self.ctc_projection = nn.Linear(self.config.d_model, self.tokenizer.vocab_size)

    def forward(
        self, waveforms: torch.Tensor, waveform_lengths: torch.Tensor
    ) -> ASROutput:
        features, frame_lengths = self.frontend(waveforms, waveform_lengths)
        if self.training and self.config.spec_augment:
            features = spec_augment(features, frame_lengths, self.config)
        encoded, output_lengths, _ = self.encoder(features, frame_lengths)
        logits = self.ctc_projection(encoded)
        return ASROutput(logits, output_lengths)
