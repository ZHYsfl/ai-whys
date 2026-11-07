from collections.abc import Sequence


def edit_distance(reference: Sequence, hypothesis: Sequence) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, ref_item in enumerate(reference, start=1):
        current = [i]
        for j, hyp_item in enumerate(hypothesis, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (ref_item != hyp_item),
                )
            )
        previous = current
    return previous[-1]


def error_counts(
    references: list[str], hypotheses: list[str], level: str
) -> tuple[int, int]:
    if len(references) != len(hypotheses) or not references:
        raise ValueError("references and hypotheses must have the same non-zero length")
    if level not in {"word", "char"}:
        raise ValueError("level must be 'word' or 'char'")
    edits = 0
    total = 0
    for reference, hypothesis in zip(references, hypotheses, strict=True):
        ref_units = reference.split() if level == "word" else list(reference.replace(" ", ""))
        hyp_units = hypothesis.split() if level == "word" else list(hypothesis.replace(" ", ""))
        edits += edit_distance(ref_units, hyp_units)
        total += len(ref_units)
    return edits, total


def error_rate(
    references: list[str], hypotheses: list[str], level: str
) -> float:
    edits, total = error_counts(references, hypotheses, level)
    if total == 0:
        return 0.0 if edits == 0 else float("inf")
    return edits / total
