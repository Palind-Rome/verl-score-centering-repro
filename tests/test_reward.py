"""Contracts of the pinned official verl scorer and routing, not a new scorer.

Set VERL_SOURCE to the pinned upstream checkout. Only standard-library upstream
modules are imported, so these tests do not require torch, vLLM, or GPU access.
"""

import importlib.util
import os
from pathlib import Path
import sys
import types

import pytest

from sc_repro.data_utils import canonical_integer
from sc_repro.diagnostics import response_diagnostics
from scripts.prepare_data import load_official_reward


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def official():
    source = Path(os.environ.get("VERL_SOURCE", ROOT.parent / "verl"))
    if not source.is_dir():
        pytest.fail("Set VERL_SOURCE to the pinned verl checkout before running reward tests")
    scorer, metadata = load_official_reward(source)
    return source, scorer, metadata


@pytest.fixture
def official_router(official, monkeypatch):
    source, _, _ = official
    # Load actual upstream modules under their normal names, bypassing only the
    # top-level verl/__init__.py (which otherwise imports torch). Do not replace
    # any routing or scoring implementation.
    for name, relative in (("verl", "verl"), ("verl.utils", "verl/utils")):
        package = types.ModuleType(name)
        package.__path__ = [str(source / relative)]
        monkeypatch.setitem(sys.modules, name, package)
    for name, relative in (
        ("verl.utils.import_utils", "verl/utils/import_utils.py"),
        ("verl.utils.reward_score", "verl/utils/reward_score/__init__.py"),
        ("verl.utils.reward_score.math_dapo", "verl/utils/reward_score/math_dapo.py"),
    ):
        spec = importlib.util.spec_from_file_location(name, source / relative)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
    return sys.modules["verl.utils.reward_score"].default_compute_score


@pytest.mark.parametrize("source", ["math_dapo", "aime2024", "aime2025"])
def test_official_default_routes_all_experiment_sources_to_signed_math_dapo(official_router, source):
    assert official_router(source, "Answer: 42", "42") == {"score": 1.0, "acc": True, "pred": "42"}
    assert official_router(source, "Answer: 43", "42") == {"score": -1.0, "acc": False, "pred": "43"}


@pytest.mark.parametrize("source", ["AIME2024", "AIME2025", "diagnostic/dapo_smoke"])
def test_nonstandard_dataset_labels_do_not_silently_select_a_reward(official_router, source):
    with pytest.raises(NotImplementedError):
        official_router(source, "Answer: 42", "42")


@pytest.mark.parametrize("text,ground_truth,score,pred", [
    ("Reasoning\nAnswer: 42", "42", 1.0, "42"),
    ("answer: $42$", "42", 1.0, "42"),
    ("Answer: 42 units", "42", 1.0, "42"),
    ("Answer: 9\nAnswer: 42", "42", 1.0, "42"),
    ("Answer: 42\nAdditional prose on another line", "42", 1.0, "42"),
    (r"Thus \boxed{42}. A short summary follows.", "42", 1.0, "42"),
    ("<think>Unfinished reasoning\nAnswer: 42", "42", 1.0, "42"),
    ("Answer: wrong\n" + r"\boxed{42}", "42", -1.0, "wrong"),
    (r"\boxed{42", "42", -1.0, "[INVALID]"),
    ("Answer: 42.0", "42", -1.0, "42.0"),
    ("Answer: 42.", "42", -1.0, "42."),
    ("Answer: 042", "42", -1.0, "042"),
    ("Answer: 1,234", "1234", 1.0, "1234"),
    (r"\boxed{1,234}", "1234", -1.0, "1,234"),
])
def test_documented_official_contract(official, text, ground_truth, score, pred):
    _, scorer, _ = official
    assert scorer(text, ground_truth) == {"score": score, "acc": score == 1.0, "pred": pred}


def test_official_answer_search_is_limited_to_last_300_characters(official):
    _, scorer, _ = official
    assert scorer("Answer: 42\n" + "x" * 280, "42")["score"] == 1.0
    assert scorer("Answer: 42\n" + "x" * 300, "42")["score"] == -1.0


def test_official_boxed_fallback_is_limited_to_last_100_characters(official):
    _, scorer, _ = official
    assert scorer(r"\boxed{42}" + "x" * 90, "42")["score"] == 1.0
    assert scorer(r"\boxed{42}" + "x" * 91, "42")["score"] == -1.0


@pytest.mark.parametrize("value", [True, 42.0, "42.0", "2+2", "1,23", None, "9" * 129])
def test_data_label_validation_rejects_noninteger_metadata(value):
    with pytest.raises(ValueError):
        canonical_integer(value)


def test_postprocessing_diagnostics_have_no_reward_output_or_effect(official):
    _, scorer, _ = official
    text = "<think>" + "Repeated sentence. " * 300 + "\nAnswer: 42"
    before = scorer(text, "42")
    diagnostics = response_diagnostics(text)
    assert "score" not in diagnostics and "acc" not in diagnostics
    assert diagnostics["explicit_unclosed_thinking"] == 1.0
    assert diagnostics["response_repetition"] > 0.8
    assert scorer(text, "42") == before == {"score": 1.0, "acc": True, "pred": "42"}
