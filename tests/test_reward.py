import math
import importlib.util
from pathlib import Path
import sys

import pytest

from sc_repro.reward import canonical_integer, compute_score, extract_final_integer, response_repetition


@pytest.mark.parametrize("text,value", [
    ("Reasoning\nAnswer: 42", 42),
    ("Answer: $042$", 42),
    ("**Answer:** -17.", -17),
    ("**Answer: 42**", 42),
    (r"Answer: \boxed{+42}", 42),
    ("Answer:\n42", 42),
    (r"Thus the result is $\boxed{42}$.", 42),
    (r"Final result: \boxed{1,234}", 1234),
    ("<think>Answer: 9</think>\nAnswer: 42", 42),
    ("Answer: 9\nWe correct that.\nAnswer: 42", 42),
])
def test_final_integer_formats(text, value):
    assert extract_final_integer(text).value == value
    assert compute_score("any", text, str(value))["score"] == 1


@pytest.mark.parametrize("text", [
    "Reasoning contains 42 but no final answer",
    "Answer: 42/1", "Answer: 42.0", "Answer: 4.2e1", "Answer: 42 or 43",
    "Answer: 42 and then more reasoning", "Answer: 4,2", "Answer: forty two",
    "<think>Answer: 42", "<think>Answer: 42</think>",
    r"We tried \boxed{42}, but that is wrong.", r"\boxed{42", r"\boxed{\frac{42}{1}}",
    "Answer: __import__('os').system('false')", "Answer: 42\nAnswer: unknown",
    r"Answer: 43 or \boxed{42}", r"\boxed{42} 43",
])
def test_refuses_ambiguous_or_incomplete_output(text):
    result = compute_score("any", text, "42")
    assert result["score"] == -1
    assert result["acc"] == 0
    assert result["parse_failure"] == 1


def test_incorrect_well_formed_is_not_parse_failure():
    result = compute_score("math_dapo", "Answer: 43", "42", extra_info={}, reward_router_address=None)
    assert result["score"] == -1
    assert result["parse_success"] == 1
    assert result["acc"] == 0
    assert all(isinstance(value, float) and math.isfinite(value) for value in result.values())


@pytest.mark.parametrize("value", [True, 42.0, "42.0", "2+2", "1,23", None, "9" * 129])
def test_bad_ground_truth_is_an_error(value):
    with pytest.raises(ValueError):
        canonical_integer(value)
    with pytest.raises(ValueError):
        compute_score("any", "Answer: 42", value)


def test_repetition_diagnostic_does_not_change_reward():
    repeated = ("This repeats a long sentence again and again. " * 300) + "\nAnswer: 42"
    result = compute_score("math_dapo", repeated, "42")
    assert result["response_repetition"] > 0.8
    assert result["score"] == 1
    assert response_repetition("Answer: 42") == 0


def test_verl_dynamic_module_loading_without_sys_modules_registration():
    # verl.utils.import_utils.load_module executes custom reward modules without
    # necessarily registering them. Dataclass string annotations must not depend
    # on an existing sys.modules entry for the dynamically generated name.
    name = "sc_custom_reward_unregistered"
    path = Path(__file__).resolve().parents[1] / "sc_repro/reward.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert name not in sys.modules
    spec.loader.exec_module(module)
    assert module.compute_score("math_dapo", "Answer: 42", "42")["score"] == 1.0
