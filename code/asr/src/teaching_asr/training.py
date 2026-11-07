import torch

from .ctc import ctc_loss
from .data import SpeechBatch
from .model import ConformerCTC


def training_step(
    model: ConformerCTC,
    batch: SpeechBatch,
    optimizer: torch.optim.Optimizer,
    grad_clip: float = 1.0,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(batch.waveforms, batch.waveform_lengths)
    loss = ctc_loss(
        output.logits,
        batch.targets,
        output.output_lengths,
        batch.target_lengths,
        model.tokenizer.blank_id,
    )
    if not torch.isfinite(loss):
        raise RuntimeError(f"non-finite CTC loss: {loss.item()}")
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    if not torch.isfinite(grad_norm):
        raise RuntimeError(f"non-finite gradient norm: {grad_norm.item()}")
    optimizer.step()
    return {"loss": float(loss.detach()), "grad_norm": float(grad_norm.detach())}
