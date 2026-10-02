"""Postprocessing diagnostics for saved rollout text; never produces a reward."""

import re


def response_diagnostics(solution_str: str) -> dict[str, float]:
    """Measure repetition and thinking markers without changing official scoring.

    Repetition is the fraction of repeated 64-character windows sampled every
    16 characters after whitespace normalization. It is a heuristic. An absent
    thinking end marker alone does not prove truncation or invalid formatting.
    """
    text = re.sub(r"\s+", " ", solution_str)
    repetition = 0.0
    if len(text) >= 256:
        windows = [text[i : i + 64] for i in range(0, len(text) - 63, 16)]
        repetition = 1.0 - len(set(windows)) / len(windows)
    return {
        "response_repetition": repetition,
        "think_start_present": float("<think>" in solution_str),
        "think_end_present": float("</think>" in solution_str),
        "explicit_unclosed_thinking": float(solution_str.rfind("<think>") > solution_str.rfind("</think>")),
    }
