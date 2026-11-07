from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from .ctc import beam_search_decode, ctc_loss
from .data import SpeechExample, collate_speech
from .metrics import error_counts
from .model import ConformerCTC


@torch.no_grad()
def evaluate_model(
    model: ConformerCTC,
    dataset: Dataset[SpeechExample],
    batch_size: int = 4,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_speech)
    model.eval()
    total_loss = 0.0
    references: list[str] = []
    hypotheses: list[str] = []
    predictions = []
    for batch in loader:
        batch = batch.to(device)
        output = model(batch.waveforms, batch.waveform_lengths)
        loss = ctc_loss(
            output.logits,
            batch.targets,
            output.output_lengths,
            batch.target_lengths,
            model.tokenizer.blank_id,
        )
        total_loss += float(loss) * batch.waveforms.shape[0]
        decoded = beam_search_decode(output.logits, output.output_lengths, model.tokenizer)
        references.extend(batch.transcripts)
        hypotheses.extend(decoded)
        for sample_id, reference, hypothesis in zip(
            batch.sample_ids, batch.transcripts, decoded, strict=True
        ):
            predictions.append(
                {"sample_id": sample_id, "reference": reference, "hypothesis": hypothesis}
            )

    word_edits, word_total = error_counts(references, hypotheses, "word")
    char_edits, char_total = error_counts(references, hypotheses, "char")
    return {
        "loss": total_loss / len(dataset),
        "wer": (
            word_edits / word_total
            if word_total
            else (0.0 if word_edits == 0 else float("inf"))
        ),
        "cer": (
            char_edits / char_total
            if char_total
            else (0.0 if char_edits == 0 else float("inf"))
        ),
        "word_edits": word_edits,
        "word_total": word_total,
        "char_edits": char_edits,
        "char_total": char_total,
        "num_samples": len(dataset),
        "predictions": predictions,
    }
