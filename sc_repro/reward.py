"""Strict, dependency-free integer reward for the SC mathematics experiment.

The preserved DAPO instructions request a final ``Answer:`` line. A final
``\\boxed{integer}`` is also accepted. There is no expression evaluation, fuzzy
matching, or search for an arbitrary earlier number in the response.
"""

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class ParsedAnswer:
    value: int | None
    format: str
    error: str = ""


_INTEGER = re.compile(r"[+-]?(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)")
_ANSWER_MARKER = re.compile(r"(?im)^[ \t]*(?P<bold>\*\*)?Answer[ \t]*:[ \t]*(?P<label_end>\*\*)?")
_BOX_MARKER = re.compile(r"\\boxed[ \t]*\{")


def canonical_integer(value: object) -> int:
    """Validate an integer label; never silently truncate floats or expressions."""
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError("Expected an integer or integer string")
    text = str(value).strip()
    if len(text) > 128 or _INTEGER.fullmatch(text) is None:
        raise ValueError("Invalid integer label")
    return int(text.replace(",", ""))


def _unwrap_integer(text: str) -> int | None:
    text = text.strip()
    # A sentence period and matched mathematical/Markdown wrappers are harmless.
    if text.endswith("."):
        text = text[:-1].rstrip()
    for _ in range(5):
        before = text
        for left, right in (("**", "**"), ("$", "$"), (r"\(", r"\)"), (r"\[", r"\]")):
            if text.startswith(left) and text.endswith(right) and len(text) >= len(left) + len(right):
                text = text[len(left) : -len(right)].strip()
                break
        else:
            if text.startswith(r"\boxed{") and text.endswith("}"):
                text = text[len(r"\boxed{") : -1].strip()
        if text == before:
            break
    try:
        return canonical_integer(text)
    except ValueError:
        return None


def extract_final_integer(solution_str: str) -> ParsedAnswer:
    """Parse the final declared answer, refusing unfinished reasoning and suffixes."""
    if not isinstance(solution_str, str):
        return ParsedAnswer(None, "none", "invalid_response")
    text = solution_str.strip()
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1].strip()
    if "<think>" in text:
        return ParsedAnswer(None, "none", "unfinished_thinking")

    answers = list(_ANSWER_MARKER.finditer(text))
    boxes = list(_BOX_MARKER.finditer(text))
    if answers:
        # A later malformed declaration must not fall back to an earlier answer.
        marker = answers[-1]
        body = text[marker.end() :].strip()
        if marker.group("bold") and not marker.group("label_end") and body.endswith("**"):
            body = body[:-2].rstrip()
        value = _unwrap_integer(body)
        if value is not None:
            return ParsedAnswer(value, "answer")
        if not boxes or boxes[-1].start() < marker.end():
            return ParsedAnswer(None, "answer", "malformed_final_answer")
        # Permit "Answer: ... \\boxed{42}" only when that entire declaration is
        # itself a boxed integer; prose/contradicting numbers remain invalid.
        return ParsedAnswer(None, "answer", "malformed_final_answer")

    if boxes:
        marker = boxes[-1]
        closing = text.find("}", marker.end())
        if closing < 0:
            return ParsedAnswer(None, "boxed", "unclosed_box")
        suffix = text[closing + 1 :].strip()
        # Allow closing math/Markdown delimiters and punctuation, but no prose
        # or second number after the final box.
        if re.fullmatch(r"(?:\$|\\\)|\\\]|\*\*|[.。!！\s])*", suffix) is None:
            return ParsedAnswer(None, "boxed", "trailing_text")
        value = _unwrap_integer(text[marker.start() : closing + 1])
        return ParsedAnswer(value, "boxed", "" if value is not None else "noninteger_box")
    return ParsedAnswer(None, "none", "missing_final_answer")


def response_repetition(solution_str: str) -> float:
    """Fraction of repeated 64-character windows, sampled every 16 characters.

    A cheap language-independent diagnostic, not a reward component. Short
    answers return zero. Whitespace runs are normalized before measurement.
    """
    text = re.sub(r"\s+", " ", solution_str)
    if len(text) < 256:
        return 0.0
    windows = [text[i : i + 64] for i in range(0, len(text) - 63, 16)]
    return 1.0 - len(set(windows)) / len(windows)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str | int,
    extra_info: dict | None = None,
    **kwargs: object,
) -> dict[str, float]:
    """verl custom-reward contract: signed score and numeric diagnostics.

    Invalid ground truth raises: dataset bugs must not silently become negative
    rewards. ``acc`` is 0/1 and is the validation accuracy; ``score`` is +/-1.
    Additional kwargs support both naive and asynchronous reward managers.
    """
    del data_source, extra_info, kwargs
    expected = canonical_integer(ground_truth)
    parsed = extract_final_integer(solution_str)
    correct = parsed.value is not None and parsed.value == expected
    valid = parsed.value is not None
    return {
        "score": 1.0 if correct else -1.0,
        "acc": float(correct),
        "parse_success": float(valid),
        "parse_failure": float(not valid),
        "answer_format": float(valid and parsed.format == "answer"),
        "boxed_format": float(valid and parsed.format == "boxed"),
        "unfinished_thinking": float(parsed.error == "unfinished_thinking"),
        "response_repetition": response_repetition(solution_str),
    }
