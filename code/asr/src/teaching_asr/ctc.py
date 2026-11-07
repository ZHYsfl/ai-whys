import math

import torch
from torch.nn import functional as F

from .text import CharTokenizer


def _required_ctc_lengths(targets: torch.Tensor, target_lengths: torch.Tensor) -> torch.Tensor:
    if targets.shape[1] == 0:
        return target_lengths.clone()
    target_positions = torch.arange(targets.shape[1] - 1, device=targets.device)
    repeated = (targets[:, 1:] == targets[:, :-1]) & (
        target_positions[None, :] < (target_lengths - 1)[:, None]
    )
    return target_lengths + repeated.long().sum(dim=1)


def validate_ctc_inputs(
    logits: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank_id: int,
) -> None:
    if logits.ndim != 3 or logits.shape[1] == 0 or logits.shape[2] < 2:
        raise ValueError(f"expected logits [B, T, V], got {tuple(logits.shape)}")
    batch_size, time_steps, vocab_size = logits.shape
    if targets.ndim != 2 or targets.shape[0] != batch_size:
        raise ValueError("targets must have shape [B, L]")
    if input_lengths.shape != (batch_size,) or target_lengths.shape != (batch_size,):
        raise ValueError("input_lengths and target_lengths must have shape [B]")
    if not 0 <= blank_id < vocab_size:
        raise ValueError("blank_id is outside the output vocabulary")
    if (input_lengths <= 0).any() or (input_lengths > time_steps).any():
        raise ValueError("input lengths must be in [1, T]")
    if (target_lengths < 0).any() or (target_lengths > targets.shape[1]).any():
        raise ValueError("target lengths must be in [0, L]")
    valid_targets = targets[lengths_to_target_mask(target_lengths, targets.shape[1])]
    if valid_targets.numel() and (
        (valid_targets < 0).any()
        or (valid_targets >= vocab_size).any()
        or (valid_targets == blank_id).any()
    ):
        raise ValueError("target ids must be non-blank ids in the output vocabulary")
    required = _required_ctc_lengths(targets, target_lengths)
    if (input_lengths < required).any():
        index = int((input_lengths < required).nonzero()[0])
        raise ValueError(
            f"sample {index} cannot be aligned by CTC: input length {int(input_lengths[index])} "
            f"< target length {int(target_lengths[index])} plus required repeated-label blanks; "
            f"minimum is {int(required[index])}"
        )


def lengths_to_target_mask(target_lengths: torch.Tensor, max_length: int) -> torch.Tensor:
    return torch.arange(max_length, device=target_lengths.device)[None, :] < target_lengths[:, None]


def ctc_loss(
    logits: torch.Tensor, # (batch_size, time_steps, vocab_size)
    targets: torch.Tensor, # (batch_size, max_target_length) 
    input_lengths: torch.Tensor, # (batch_size,)
    target_lengths: torch.Tensor, # (batch_size,)
    blank_id: int = 0,
    reduction: str = "mean",
) -> torch.Tensor:
    if reduction not in {"mean", "sum", "none"}:
        raise ValueError("reduction must be 'mean', 'sum' or 'none'")
    validate_ctc_inputs(logits, targets, input_lengths, target_lengths, blank_id)
    log_probs = F.log_softmax(logits.float(), dim=-1).transpose(0, 1) # (time_steps, batch_size, vocab_size)
    return F.ctc_loss(
        log_probs, # 每个音频帧输出每个 token 的对数概率
        targets, # 正确的字符序列
        input_lengths, # 音频的有效长度
        target_lengths, # 文字的有效长度
        blank=blank_id,
        reduction=reduction,
        zero_infinity=True, # 如果正确文字的总概率接近 0，损失就可能变成无穷大，进而污染整个训练过程。所以遇到无穷大的 CTC 损失时，将其及对应梯度归零，避免一个异常样本破坏训练。
    ) # 0D if reduction == "mean", the loss is averaged over the batch; 0D if reduction == "sum", the loss is summed over the batch; 1D (batch_size,) if reduction == "none", the loss is returned for each sample in the batch.


def teaching_ctc_log_probability(
    log_probs: torch.Tensor, target: torch.Tensor, blank_id: int = 0
) -> torch.Tensor:
    if log_probs.ndim != 2 or target.ndim != 1:
        raise ValueError("expected one log_probs [T, V] and one target [U]")
    if log_probs.shape[0] == 0:
        raise ValueError("log_probs cannot have zero time steps")
    target_length = target.shape[0]
    neg_inf = -1e30
    extended = torch.full(
        (2 * target_length + 1,), blank_id, dtype=torch.long, device=target.device
    )
    extended[1::2] = target
    emissions = log_probs[:, extended]
    alpha = log_probs.new_full((extended.shape[0],), neg_inf)
    alpha[0] = emissions[0, 0]
    if target_length:
        alpha[1] = emissions[0, 1]

    for time in range(1, log_probs.shape[0]):
        stay = alpha
        advance_one = F.pad(alpha[:-1], (1, 0), value=neg_inf)
        if alpha.shape[0] >= 2:
            advance_two = F.pad(alpha[:-2], (2, 0), value=neg_inf)
        else:
            advance_two = log_probs.new_full(alpha.shape, neg_inf)
        can_skip_blank = torch.zeros_like(extended, dtype=torch.bool)
        can_skip_blank[2:] = (extended[2:] != blank_id) & (extended[2:] != extended[:-2])
        advance_two = advance_two.masked_fill(~can_skip_blank, neg_inf)
        alpha = torch.logsumexp(torch.stack((stay, advance_one, advance_two)), dim=0)
        alpha = alpha + emissions[time]

    if target_length == 0:
        return alpha[0]
    return torch.logaddexp(alpha[-1], alpha[-2])


def _validate_decode_inputs(
    logits: torch.Tensor,
    input_lengths: torch.Tensor,
    tokenizer: CharTokenizer,
) -> None:
    if logits.ndim != 3 or logits.shape[2] != tokenizer.vocab_size:
        raise ValueError(f"expected logits [B, T, {tokenizer.vocab_size}]")
    if input_lengths.shape != (logits.shape[0],):
        raise ValueError("input_lengths must have shape [B]")
    if (input_lengths <= 0).any() or (input_lengths > logits.shape[1]).any():
        raise ValueError("input lengths must be in [1, T]")


def greedy_decode(
    logits: torch.Tensor, # (batch_size, time_steps, vocab_size)
    input_lengths: torch.Tensor, # (batch_size,)
    tokenizer: CharTokenizer,
) -> list[str]:
    _validate_decode_inputs(logits, input_lengths, tokenizer)
    best_paths = logits.argmax(dim=-1).cpu() # (batch_size, time_steps)
    lengths = input_lengths.cpu() # (batch_size,)
    transcripts = []
    for path, length in zip(best_paths, lengths, strict=True):
        output_ids = []
        previous = tokenizer.blank_id
        for token in path[: int(length)].tolist():
            if token != tokenizer.blank_id and token != previous:
                output_ids.append(token)
            previous = token
        transcripts.append(tokenizer.decode(output_ids))
    return transcripts


def _log_add_exp(first: float, second: float) -> float:
    if first == -math.inf:
        return second
    if second == -math.inf:
        return first
    largest = max(first, second)
    return largest + math.log1p(math.exp(-abs(first - second))) # 为什么不直接写 log(exp(first) + exp(second))？因为对数概率可能非常小，直接 exp 容易下溢成 0，所以使用稳定形式。


@torch.no_grad()
def beam_search_decode(
    logits: torch.Tensor, # (batch_size, time_steps, vocab_size)
    input_lengths: torch.Tensor, # (batch_size,)
    tokenizer: CharTokenizer,
    beam_size: int = 8,
) -> list[str]:
    _validate_decode_inputs(logits, input_lengths, tokenizer)
    if beam_size < 1:
        raise ValueError("beam_size must be >= 1")
    log_probs = F.log_softmax(logits.float(), dim=-1).cpu() # # (batch_size, time_steps, vocab_size)
    lengths = input_lengths.cpu() # (batch_size,)
    transcripts = []
    neg_inf = -math.inf

    for sample_log_probs, length in zip(log_probs, lengths, strict=True):
        beams: dict[tuple[int, ...], tuple[float, float]] = {(): (0.0, neg_inf)}
        next_beams: dict[tuple[int, ...], tuple[float, float]] = {}

        def update(
            prefix: tuple[int, ...], blank_ending: bool, score: float
        ) -> None:
            blank_score, nonblank_score = next_beams.get(prefix, (neg_inf, neg_inf))
            if blank_ending:
                blank_score = _log_add_exp(blank_score, score)
            else:
                nonblank_score = _log_add_exp(nonblank_score, score)
            next_beams[prefix] = (blank_score, nonblank_score)

        for frame in sample_log_probs[: int(length)].tolist():
            next_beams = {}
            for prefix, (blank_score, nonblank_score) in beams.items():
                total_score = _log_add_exp(blank_score, nonblank_score)
                update(prefix, True, total_score + frame[tokenizer.blank_id]) # 情况一：当前帧输出 blank
                for token, token_log_prob in enumerate(frame):
                    if token == tokenizer.blank_id:
                        continue
                    if prefix and prefix[-1] == token: # 情况三：输出与上一字符相同的非 blank
                        update(prefix, False, nonblank_score + token_log_prob) # 3.1 上一帧不是 blank：重复输出，不增加字符
                        update(prefix + (token,), False, blank_score + token_log_prob) # 3.2 上一帧是 blank：这是新字符，文本变成 "aa"
                    else: #  情况二：输出一个与上一字符不同的非 blank
                        update(prefix + (token,), False, total_score + token_log_prob)
            beams = dict(
                sorted(
                    next_beams.items(),
                    key=lambda item: _log_add_exp(item[1][0], item[1][1]),
                    reverse=True,
                )[:beam_size]
            )

        best_prefix = max(
            beams, key=lambda prefix: _log_add_exp(*beams[prefix])
        )
        transcripts.append(tokenizer.decode(best_prefix))
    return transcripts # list[str] 大小为 batch_size
