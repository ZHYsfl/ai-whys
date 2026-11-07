from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from .data import SpeechBatch, SpeechExample, collate_speech
from .metrics import error_counts
from .qwen3_asr import Qwen3ASR


def llm_training_step(
    model: Qwen3ASR,
    batch: SpeechBatch,
    optimizer: torch.optim.Optimizer,
    grad_clip: float = 1.0,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(batch.waveforms, batch.waveform_lengths, batch.transcripts)
    if not torch.isfinite(output.loss):
        raise RuntimeError(f"non-finite LLM loss: {output.loss.item()}")
    output.loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    if not torch.isfinite(grad_norm):
        raise RuntimeError(f"non-finite gradient norm: {grad_norm.item()}")
    optimizer.step()
    return {"loss": float(output.loss.detach()), "grad_norm": float(grad_norm.detach())}


@torch.no_grad()
def evaluate_llm(
    model: Qwen3ASR,
    dataset: Dataset[SpeechExample],
    batch_size: int = 2,
    device: str | torch.device = "cpu",
    generate: bool = True,
    max_new_tokens: int = 128,
) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_speech)
    model.eval()
    total_loss = 0.0
    total_target_tokens = 0
    references: list[str] = []
    hypotheses: list[str] = []
    predictions = []
    for batch in loader:
        batch = batch.to(device)
        output = model(batch.waveforms, batch.waveform_lengths, batch.transcripts)
        target_tokens = sum(
            len(model.tokenizer.encode(transcript)) + 1 for transcript in batch.transcripts
        )
        total_loss += float(output.loss) * target_tokens
        total_target_tokens += target_tokens
        if generate:
            decoded = model.transcribe(
                batch.waveforms, batch.waveform_lengths, max_new_tokens=max_new_tokens
            )
            references.extend(batch.transcripts)
            hypotheses.extend(decoded)
            for sample_id, reference, hypothesis in zip(
                batch.sample_ids, batch.transcripts, decoded, strict=True
            ):
                predictions.append(
                    {"sample_id": sample_id, "reference": reference, "hypothesis": hypothesis}
                )

    result = {"loss": total_loss / total_target_tokens, "num_samples": len(dataset)}
    if generate:
        word_edits, word_total = error_counts(references, hypotheses, "word")
        char_edits, char_total = error_counts(references, hypotheses, "char")
        result.update(
            {
                "wer": word_edits / word_total if word_total else 0.0,
                "cer": char_edits / char_total if char_total else 0.0,
                "word_edits": word_edits,
                "word_total": word_total,
                "char_edits": char_edits,
                "char_total": char_total,
                "predictions": predictions,
            }
        )
    return result
