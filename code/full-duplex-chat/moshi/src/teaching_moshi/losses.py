# Derived from kyutai-labs/moshi-finetune (Apache-2.0); see NOTICE and LICENSES/.

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .model import LMOutput, MoshiLM


def compute_loss_with_mask(
    logits: torch.Tensor, # text : (batch_size, 1, new_num_frames-1, 260) audio: (batch_size, 8, new_num_frames-1, 64)
    target: torch.Tensor, # text : (batch_size, 1, new_num_frames-1) audio : (batch_size, 8, new_num_frames-1)
    target_mask: torch.Tensor, # text : (batch_size, 1, new_num_frames-1) audio : (batch_size, 8, new_num_frames-1)
    first_codebook_weight_multiplier: float = 1.0, # text : 1.0 audio : 100.0
    text_padding_weight: float = 1.0, # text : 0.5 audio : 1.0
    text_padding_ids: set[int] | None = None, # text : {3, 0} audio : None
) -> torch.Tensor:
    """Weighted CE over [B,K,T], with invalid delay/padding positions removed."""
    if logits.shape[:-1] != target.shape or target.shape != target_mask.shape:
        raise ValueError("logits, target and mask shapes are inconsistent")
    safe_target = torch.where(target_mask, target, torch.zeros_like(target))
    weights = target_mask.to(logits.dtype)
    weights[:, 0] *= first_codebook_weight_multiplier
    if text_padding_ids is not None:
        for token_id in text_padding_ids:
            weights[safe_target == token_id] *= text_padding_weight
    loss = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        safe_target.reshape(-1),
        reduction="none",
    ) # (batch_size * 1 * new_num_frames-1,)
    weighted = torch.where(weights.reshape(-1) > 0, loss * weights.reshape(-1), 0.0) # (batch_size * 1 * new_num_frames-1,)
    return weighted.sum() / weights.sum().clamp_min(1.0)


@dataclass
class MoshiLossOutput:
    total: torch.Tensor
    text: torch.Tensor
    audio: torch.Tensor


def moshi_loss(
    model: MoshiLM,
    codes: torch.Tensor, # (batch_size, 17, time_steps)
    output: LMOutput,
    first_codebook_weight_multiplier: float = 100.0,
    text_padding_weight: float = 0.5,
) -> MoshiLossOutput:
    text = compute_loss_with_mask(
        output.text_logits, # (batch_size, 1, new_num_frames-1, 260)
        codes[:, :1], # (batch_size, 1, new_num_frames-1)
        output.text_mask, # (batch_size, 1, new_num_frames-1)
        text_padding_weight=text_padding_weight, # 0.5
        text_padding_ids={
            model.text_padding_token_id, # 3
            model.end_of_text_padding_id, # 0
        },
    )
    audio = compute_loss_with_mask(
        output.logits, # (batch_size, 8, new_num_frames-1, 64)
        codes[:, 1 : model.dep_q + 1], # (batch_size, 8, new_num_frames-1)
        output.mask, # (batch_size, 8, new_num_frames-1)
        first_codebook_weight_multiplier=first_codebook_weight_multiplier, # 100
    )
    return MoshiLossOutput(text + audio, text, audio)
