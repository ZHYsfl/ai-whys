import torch

from .aut import AuT
from .data import SpeechBatch
from .llm_text import LanguageTokenizer, collate_language_targets


def aut_training_step(
    model: AuT,
    batch: SpeechBatch,
    tokenizer: LanguageTokenizer,
    optimizer: torch.optim.Optimizer,
    grad_clip: float = 1.0,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    target_ids, target_lengths = collate_language_targets(batch, tokenizer)
    output = model(batch.waveforms, batch.waveform_lengths, target_ids, target_lengths)
    if not torch.isfinite(output.loss):
        raise RuntimeError(f"non-finite AuT loss: {output.loss.item()}")
    output.loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    if not torch.isfinite(grad_norm):
        raise RuntimeError(f"non-finite gradient norm: {grad_norm.item()}")
    optimizer.step()
    return {"loss": float(output.loss.detach()), "grad_norm": float(grad_norm.detach())}


@torch.no_grad()
def evaluate_aut(
    model: AuT,
    batch: SpeechBatch,
    tokenizer: LanguageTokenizer,
) -> float:
    model.eval()
    target_ids, target_lengths = collate_language_targets(batch, tokenizer)
    return float(model(batch.waveforms, batch.waveform_lengths, target_ids, target_lengths).loss)
