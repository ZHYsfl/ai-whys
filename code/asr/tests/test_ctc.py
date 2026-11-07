import itertools

import pytest
import torch

from teaching_asr.ctc import (
    beam_search_decode,
    ctc_loss,
    greedy_decode,
    teaching_ctc_log_probability,
)
from teaching_asr.text import CharTokenizer


def test_teaching_ctc_matches_pytorch_with_repeated_label() -> None:
    torch.manual_seed(0)
    logits = torch.randn(1, 5, 4, requires_grad=True)
    targets = torch.tensor([[1, 1]])
    input_lengths = torch.tensor([5])
    target_lengths = torch.tensor([2])
    expected = ctc_loss(logits, targets, input_lengths, target_lengths, reduction="none")[0]
    actual = -teaching_ctc_log_probability(logits[0].log_softmax(-1), targets[0])
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
    actual.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_teaching_ctc_matches_brute_force_paths() -> None:
    torch.manual_seed(1)
    log_probs = torch.randn(3, 3).log_softmax(dim=-1)
    target = torch.tensor([1, 1])
    total_probability = torch.tensor(0.0)
    for path in itertools.product(range(3), repeat=3):
        collapsed = []
        previous = 0
        for token in path:
            if token != 0 and token != previous:
                collapsed.append(token)
            previous = token
        if collapsed == target.tolist():
            path_probability = sum(log_probs[time, token] for time, token in enumerate(path))
            total_probability = total_probability + path_probability.exp()
    expected = total_probability.clamp_min(1e-30).log()
    actual = teaching_ctc_log_probability(log_probs, target)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_teaching_ctc_empty_target() -> None:
    log_probs = torch.randn(3, 4).log_softmax(dim=-1)
    target = torch.empty(0, dtype=torch.long)
    actual = -teaching_ctc_log_probability(log_probs, target)
    expected = ctc_loss(
        log_probs[None], target[None], torch.tensor([3]), torch.tensor([0])
    )
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_greedy_decode_collapses_repeats_before_removing_blanks() -> None:
    tokenizer = CharTokenizer(("a", "b", "c"))
    path = [1, 1, 0, 1, 2, 2, 0]
    logits = torch.full((1, len(path), 4), -10.0)
    for time, token in enumerate(path):
        logits[0, time, token] = 10.0
    assert greedy_decode(logits, torch.tensor([len(path)]), tokenizer) == ["aab"]


def test_beam_search_can_merge_paths_beyond_greedy() -> None:
    tokenizer = CharTokenizer(("a", "b"))
    probabilities = torch.tensor([[[0.4, 0.31, 0.29], [0.4, 0.31, 0.29]]])
    logits = probabilities.log()
    assert greedy_decode(logits, torch.tensor([2]), tokenizer) == [""]
    assert beam_search_decode(logits, torch.tensor([2]), tokenizer, beam_size=3) == ["a"]


def test_beam_search_matches_brute_force_best_text() -> None:
    torch.manual_seed(2)
    tokenizer = CharTokenizer(("a", "b", "c"))
    logits = torch.randn(1, 4, 4)
    log_probs = logits[0].log_softmax(dim=-1)
    total_probabilities = {}
    for path in itertools.product(range(4), repeat=4):
        collapsed = []
        previous = 0
        for token in path:
            if token != 0 and token != previous:
                collapsed.append(token)
            previous = token
        path_log_probability = sum(
            log_probs[time, token] for time, token in enumerate(path)
        )
        key = tuple(collapsed)
        total_probabilities[key] = total_probabilities.get(key, 0.0) + float(
            path_log_probability.exp()
        )
    expected = max(total_probabilities, key=total_probabilities.get)
    actual = beam_search_decode(
        logits, torch.tensor([4]), tokenizer, beam_size=1000
    )
    assert actual == [tokenizer.decode(expected)]


def test_beam_search_rejects_empty_beam() -> None:
    with pytest.raises(ValueError, match="beam_size"):
        beam_search_decode(
            torch.randn(1, 2, 4),
            torch.tensor([2]),
            CharTokenizer(("a", "b", "c")),
            beam_size=0,
        )


def test_ctc_rejects_impossible_repeated_target() -> None:
    with pytest.raises(ValueError, match="repeated-label blanks"):
        ctc_loss(
            torch.randn(1, 2, 4),
            torch.tensor([[1, 1]]),
            torch.tensor([2]),
            torch.tensor([2]),
        )


def test_ctc_rejects_blank_in_target() -> None:
    with pytest.raises(ValueError, match="non-blank"):
        ctc_loss(
            torch.randn(1, 3, 4),
            torch.tensor([[0]]),
            torch.tensor([3]),
            torch.tensor([1]),
        )
