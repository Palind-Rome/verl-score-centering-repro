#!/usr/bin/env python3
"""Materialize locked DAPO/AIME data and an auditable verl conversion.

Requires pyarrow and huggingface_hub. Downloads only the three locked small
data files. Published DAPO rows remain unchanged by default; duplicate and
conflicting-answer candidates are reported, never silently adjudicated.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import subprocess
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sc_repro.data_utils import canonical_integer  # noqa: E402


PREFIX = (
    "Solve the following math problem step by step. The last line of your response should be "
    "of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n"
)
SUFFIX = '\n\nRemember to put your answer on its own line after "Answer:".'

# Fixed source row numbers in the locked revision, selected for short arithmetic
# questions, not for observed model accuracy. They exclude audit conflict groups.
# These are diagnostics only; they do not define or replace the formal train set.
SMOKE_SOURCE_ROWS = [
    3312, 9058, 9101, 8020, 1768, 8392, 8997, 9486,
    10455, 354, 3535, 9175, 1464, 2317, 3389, 3589,
    8625, 7974, 9027, 10882, 15060, 1353, 4350, 10773,
    14880, 1538, 4690, 6107, 9808, 10427, 1568, 2676,
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_info(path: Path) -> dict:
    data = path.read_bytes()
    return {"bytes": len(data), "sha256": sha256_bytes(data)}


def load_official_reward(upstream: Path):
    """Load the unchanged pinned math_dapo module without importing torch/verl.

    Both the score module and its default router must match the pinned commit's
    tracked bytes before execution. Only the standalone score module is loaded.
    """
    upstream = upstream.resolve()
    expected = json.loads((ROOT / "upstream.json").read_text())["commit"]
    actual = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if actual != expected:
        raise ValueError(f"Expected upstream {expected}, found {actual}")
    metadata = {"upstream_commit": actual}
    for kind, relative in (("reward", "verl/utils/reward_score/math_dapo.py"),
                           ("router", "verl/utils/reward_score/__init__.py")):
        path = upstream / relative
        tracked = subprocess.check_output(["git", "-C", str(upstream), "show", f"{actual}:{relative}"])
        if path.read_bytes() != tracked:
            raise ValueError(f"Official {kind} source differs from the pinned commit: {path}")
        metadata[f"{kind}_relative_path"] = relative
        metadata[f"{kind}_sha256"] = sha256_bytes(tracked)
    path = upstream / metadata["reward_relative_path"]
    spec = importlib.util.spec_from_file_location("_sc_official_math_dapo", path)
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module.compute_score, metadata


def normalize_problem(text: str, method: str) -> str:
    """Normalization is for audits; aggressive variants do not authorize removal."""
    if method == "raw":
        return text
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if method == "strip":
        return text.strip()
    text = unicodedata.normalize("NFKC" if method.startswith("nfkc") else "NFC", text)
    if method.endswith("no_whitespace"):
        return re.sub(r"\s+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def audit_duplicates(rows: list[dict], method: str) -> dict:
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[normalize_problem(row["prompt"], method)].append(i)
    duplicates = [indices for indices in groups.values() if len(indices) > 1]
    conflicts = []
    for indices in duplicates:
        if len({canonical_integer(rows[i]["solution"]) for i in indices}) > 1:
            conflicts.append({
                "source_rows": indices,
                "examples": [
                    {"source_row": i, "source_index": rows[i]["extra_info"]["index"],
                     "problem": rows[i]["prompt"], "answer": rows[i]["solution"]}
                    for i in indices
                ],
            })
    return {
        "duplicate_groups": len(duplicates),
        "extra_duplicate_rows": sum(len(indices) - 1 for indices in duplicates),
        "duplicate_source_rows": duplicates,
        "conflicting_groups": len(conflicts),
        "conflicting_rows": sum(len(group["source_rows"]) for group in conflicts),
        "conflicts": conflicts,
    }


def validate_dapo(rows: list[dict], official_score) -> None:
    for i, row in enumerate(rows):
        expected = canonical_integer(row["reward_model"]["ground_truth"])
        if expected != canonical_integer(row["solution"]):
            raise ValueError(f"DAPO row {i}: solution and reward ground truth disagree")
        messages = row["source_prompt"]
        if not isinstance(messages, list) or len(messages) != 1:
            raise ValueError(f"DAPO row {i}: unexpected chat structure")
        if messages[0].get("role") != "user" or not isinstance(messages[0].get("content"), str):
            raise ValueError(f"DAPO row {i}: invalid user message")
        if row["prompt"] not in messages[0]["content"]:
            raise ValueError(f"DAPO row {i}: source_prompt does not contain original problem")
        if "Answer:" not in messages[0]["content"]:
            raise ValueError(f"DAPO row {i}: unexpected answer instructions")
        if official_score(f"Answer: {expected}", str(expected))["acc"] != 1.0:
            raise ValueError(f"DAPO row {i}: gold-answer reward roundtrip failed")


def to_verl(row: dict, source_row: int, source: dict, name: str) -> dict:
    is_dapo = name == "dapo"
    problem = row["prompt"] if is_dapo else row["problem"]
    answer = row["reward_model"]["ground_truth"] if is_dapo else row["answer"]
    ground_truth = str(canonical_integer(answer))
    if not is_dapo and not 0 <= int(ground_truth) <= 999:
        raise ValueError(f"{name} row {source_row}: expected a three-digit AIME integer")
    prompt = row["source_prompt"] if is_dapo else [{"role": "user", "content": PREFIX + problem + SUFFIX}]
    return {
        "data_source": "math_dapo" if is_dapo else {"aime24": "aime2024", "aime25": "aime2025"}[name],
        "prompt": prompt,
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": {
            "split": "train" if is_dapo else "test",
            "index": str(row["extra_info"]["index"] if is_dapo else row["id"]),
            "source_row": source_row,
            "source_repo": source["repo_id"],
            "source_revision": source["revision"],
            "source_split": source["split"],
            "original_ground_truth": str(answer),
            "problem_sha256": sha256_bytes(problem.encode()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True, help="Checkout at the commit in upstream.json")
    parser.add_argument("--raw-dir", type=Path, help="Default: OUTPUT/raw; files are stored under source keys")
    parser.add_argument("--source-lock", type=Path, default=ROOT / "data/sources.lock.json")
    parser.add_argument("--offline", action="store_true", help="Require all locked files under raw-dir; no network")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    raw_dir = (args.raw_dir or output / "raw").resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_lock = json.loads(args.source_lock.read_text())
    sources = source_lock["sources"]
    official_score, official_reward = load_official_reward(args.upstream)

    import pyarrow as pa
    import pyarrow.parquet as pq

    loaded = {}
    for name, spec in sources.items():
        path = raw_dir / name / spec["filename"]
        if not path.is_file():
            if args.offline:
                raise FileNotFoundError(f"Missing locked input: {path}")
            from huggingface_hub import hf_hub_download
            hf_hub_download(
                repo_id=spec["repo_id"], repo_type="dataset", revision=spec["revision"],
                filename=spec["filename"], local_dir=raw_dir / name,
            )
        actual = file_info(path)
        if actual != {key: spec[key] for key in ("bytes", "sha256")}:
            raise ValueError(f"Locked checksum mismatch for {name}: {actual}")
        rows = (pq.read_table(path).to_pylist() if path.suffix == ".parquet" else
                [json.loads(line) for line in path.read_text().splitlines() if line.strip()])
        if len(rows) != spec["rows"]:
            raise ValueError(f"{name}: expected {spec['rows']} rows, found {len(rows)}")
        loaded[name] = rows

    dapo = loaded["dapo"]
    validate_dapo(dapo, official_score)
    label_changes = {}
    for name, rows in loaded.items():
        changes = []
        for i, row in enumerate(rows):
            original = str(row["reward_model"]["ground_truth"] if name == "dapo" else row["answer"])
            canonical = str(canonical_integer(original))
            if original != canonical:
                changes.append({"source_row": i, "original": original, "canonical": canonical})
        label_changes[name] = changes
    methods = ("raw", "strip", "nfc_space_collapse", "nfkc_space_collapse", "nfc_no_whitespace")
    audit = {method: audit_duplicates(dapo, method) for method in methods}
    source_prompts = defaultdict(list)
    for i, row in enumerate(dapo):
        source_prompts[json.dumps(row["source_prompt"], sort_keys=True, ensure_ascii=False)].append(i)
    audit["source_prompt_exact"] = {
        "duplicate_groups": sum(len(indices) > 1 for indices in source_prompts.values()),
        "extra_duplicate_rows": sum(len(indices) - 1 for indices in source_prompts.values()),
        "duplicate_source_rows": [indices for indices in source_prompts.values() if len(indices) > 1]
    }
    audit["policy"] = (
        "All published DAPO rows are retained except exact stripped eval overlap. "
        "Normalized duplicate/conflict candidates are audit information only; no answer is guessed. "
        "NFKC and no-whitespace variants can merge distinct mathematical text and do not justify deletion."
    )

    evaluations = {}
    for name in ("aime24", "aime25"):
        rows = loaded[name]
        keys = [normalize_problem(row["problem"], "strip") for row in rows]
        if len(set(keys)) != 30 or len({str(row["id"]) for row in rows}) != 30:
            raise ValueError(f"{name}: expected 30 distinct problems and ids")
        evaluations[name] = [to_verl(row, i, sources[name], name) for i, row in enumerate(rows)]

    overlap = {}
    for method in methods:
        lookup = defaultdict(list)
        for name in ("aime24", "aime25"):
            for i, row in enumerate(loaded[name]):
                lookup[normalize_problem(row["problem"], method)].append({"dataset": name, "source_row": i})
        overlap[method] = [
            {"train_source_row": i, "evaluation_matches": lookup[normalize_problem(row["prompt"], method)]}
            for i, row in enumerate(dapo) if normalize_problem(row["prompt"], method) in lookup
        ]
    excluded = {item["train_source_row"] for item in overlap["strip"]}
    train = [to_verl(row, i, sources["dapo"], "dapo") for i, row in enumerate(dapo) if i not in excluded]
    conflict_indices = {i for group in audit["nfc_no_whitespace"]["conflicts"] for i in group["source_rows"]}
    if set(SMOKE_SOURCE_ROWS) & (conflict_indices | excluded):
        raise ValueError("A smoke question is a conflict candidate or evaluation overlap")
    smoke_train = [to_verl(dapo[i], i, sources["dapo"], "dapo") for i in SMOKE_SOURCE_ROWS]
    if len({normalize_problem(dapo[i]["prompt"], "nfc_no_whitespace") for i in SMOKE_SOURCE_ROWS}) != len(SMOKE_SOURCE_ROWS):
        raise ValueError("Smoke subset contains normalized duplicate questions")
    # Smoke validation deliberately reuses eight training questions to cheaply
    # validate generation and reward plumbing. It is not a held-out benchmark.
    smoke_val = json.loads(json.dumps(smoke_train[:8]))
    for row in smoke_val:
        row["data_source"] = "math_dapo"
        row["extra_info"]["split"] = "smoke_diagnostic_train_overlap"

    output_rows = {"train": train, **evaluations, "smoke_train": smoke_train, "smoke_val": smoke_val}
    for name, rows in output_rows.items():
        for row in rows:
            answer = row["reward_model"]["ground_truth"]
            for gold_response in (f"Answer: {answer}", rf"\boxed{{{answer}}}"):
                result = official_score(gold_response, answer)
                if result["score"] != 1.0 or result["acc"] is not True:
                    raise ValueError(f"Official reward gold round-trip failed in {name}: {row['extra_info']['index']}")
    outputs = {}
    for name, rows in output_rows.items():
        path = output / f"{name}.parquet"
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, path, compression="zstd")
        if pq.read_table(path).to_pylist() != rows:
            raise AssertionError(f"Round-trip mismatch in {path}")
        outputs[path.name] = {"rows": len(rows), **file_info(path)}
    report_path = output / "audit.json"
    report_path.write_text(json.dumps({"duplicates": audit, "train_eval_overlap": overlap}, indent=2, ensure_ascii=False) + "\n")
    manifest = {
        "schema_version": 1,
        "sources": sources,
        "conversion": {
            "script_sha256": file_info(Path(__file__))["sha256"],
            "data_utils_sha256": file_info(ROOT / "sc_repro/data_utils.py")["sha256"],
            **official_reward,
            "pyarrow_version": pa.__version__,
            "train_prompt": "Original source_prompt messages are preserved exactly",
            "validation_prompt": "Original problem with the same DAPO Answer instruction wrapper",
            "train_policy": audit["policy"],
            "train_removed_for_eval_overlap": sorted(excluded),
            "reward": "Unmodified verl default_compute_score -> math_dapo.compute_score; score +/-1, acc boolean, pred string",
            "custom_reward_function": None,
            "reward_policy": (
                "Official math_dapo examines the last 300 characters, extracts the last case-insensitive Answer: "
                "match through the end of that line and applies its built-in normalization. If the resulting "
                "prediction is [INVALID], it falls back to the last balanced boxed expression in the last 100 "
                "characters, compared exactly to ground truth. A present but wrong Answer prediction blocks boxed "
                "fallback. No extra integer parsing, thinking-format gate, suffix rule, or overlong penalty is added."
            ),
            "reward_routes": {"math_dapo": "math_dapo", "aime2024": "math_dapo", "aime2025": "math_dapo"},
            "gold_roundtrip": "Every output label checked against official Answer and boxed scoring",
            "ground_truth_normalization": {
                "rule": "Integer label strings are canonicalized using str(int(label)); original_ground_truth is retained in extra_info",
                "changes": label_changes,
                "limitation": (
                    "Official scoring compares its normalized prediction as a string and does not remove leading zeroes. "
                    "For example, prediction 025 and canonical target 25 do not match. No extra scorer rule is added."
                ),
            },
            "thinking_policy": {
                "description": (
                    "No custom thinking-format scoring rules. Optional text diagnostics are computed from saved "
                    "rollouts after training and never change the official reward."
                ),
                "qwen3_tokenizer_observation": (
                    "Existing Qwen3-4B tokenizer_config marks token 151667 <think> and token 151668 </think> "
                    "as special=false. The generation template with enable_thinking=true ends with assistant newline "
                    "without a prefilled <think>; false prefills an empty closed thinking block."
                ),
                "observed_decode_probe": {
                    "transformers_version": "4.57.6",
                    "method": "AutoTokenizer from existing local Qwen3-4B; local_files_only=true; skip_special_tokens=true",
                    "cases": [
                        {"input_ids": [151667, 151668], "decoded": "<think></think>"},
                        {"input_ids": [151644, 151667, 151668, 151645], "decoded": "<think></think>"},
                    ],
                    "scope": "Observed once on the experiment's existing tokenizer; conversion does not rerun this tokenizer probe",
                },
            },
        },
        "outputs": outputs,
        "audit": {"filename": report_path.name, **file_info(report_path),
                  "summary": {key: {k: v for k, v in value.items() if isinstance(v, int)}
                              for key, value in audit.items() if isinstance(value, dict)},
                  "train_eval_overlap_counts": {method: len(items) for method, items in overlap.items()}},
        "smoke": {
            "source_rows": SMOKE_SOURCE_ROWS,
            "train_rows": len(smoke_train), "validation_rows": len(smoke_val),
            "selection": "Fixed short arithmetic questions, excludes conflict candidates; no measured model-accuracy selection",
            "validation_warning": "Eight training questions are reused for plumbing diagnostics; not held-out accuracy",
            "limitation": "No guarantee of mixed within-group rewards; inspect effective-group fraction before a learning run",
        },
        "limitations": [
            "Published data contains answer-conflict candidates that are reported but preserved by default pending review",
            "Exact normalized overlap checks cannot rule out paraphrases, translations, or base-model pretraining contamination",
            "Dataset bytes are not proven identical to the old private experiment's parquet files",
            "MATH500 is outside this initial integer-label dataset selection; official math_dapo scoring itself is not restricted to integers",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"outputs": {name: details["rows"] for name, details in outputs.items()},
                      "audit": manifest["audit"]["summary"],
                      "overlap": manifest["audit"]["train_eval_overlap_counts"]}, indent=2))


if __name__ == "__main__":
    main()
