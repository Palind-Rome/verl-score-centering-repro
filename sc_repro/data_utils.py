"""Data-label validation only. Training rewards come directly from verl."""

import re

_INTEGER = re.compile(r"[+-]?(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)")


def canonical_integer(value: object) -> int:
    """Validate a dataset label without evaluating expressions or truncating floats."""
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError("Expected an integer or integer string")
    text = str(value).strip()
    if len(text) > 128 or _INTEGER.fullmatch(text) is None:
        raise ValueError("Invalid integer label")
    return int(text.replace(",", ""))
