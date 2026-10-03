"""Keep the union of three latest and two best complete checkpoints, without copies."""
from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
import time

SCORE_KEYS = ("val-core/aime2024/acc/mean@8", "val-core/aime2025/acc/mean@8")


def complete_checkpoint(path: Path) -> bool:
    try:
        if int((path.parent/'latest_checkpointed_iteration.txt').read_text().strip()) < checkpoint_step(path):
            return False
    except (FileNotFoundError, ValueError):
        return False
    required = [path / "actor" / f"{kind}_world_size_4_rank_{rank}.pt"
                for kind in ("model", "optim", "extra_state") for rank in range(4)]
    required += [path / "data.pt", path / "transfer_queue/controller_state.pkl",
                 path / "transfer_queue/metadata.json", path / "transfer_queue/simple_storage/storage_unit_info.json"]
    required += list((path / "transfer_queue/simple_storage").glob("su_*.pkl"))
    return (len(required) == 24 and all(p.is_file() and not p.is_symlink() and p.stat().st_size > 0
                                     for p in required))


def checkpoint_step(path: Path) -> int:
    if not path.name.startswith("global_step_"):
        raise ValueError("Expected global_step_N checkpoint directory")
    return int(path.name[len("global_step_"):])


def read_metrics(path: Path) -> dict[int, dict]:
    rows = {}
    if path.exists():
        for line in path.read_bytes().splitlines(keepends=True):
            if not line.endswith(b"\n"):
                continue
            row = json.loads(line)
            rows[row["step"]] = row["data"]
    return rows


def lineage(run: Path) -> list[Path]:
    run = run.resolve()
    runs_root = run.parent
    initial = json.loads((run / "launch.json").read_text())
    result = []
    seen = set()
    while run not in seen:
        if run.parent != runs_root:
            raise ValueError("Checkpoint lineage must stay inside the experiment runs directory")
        seen.add(run)
        meta = json.loads((run / "launch.json").read_text())
        if any(meta[k] != initial[k] for k in ("method", "seed", "dataset_manifest_sha256")):
            raise ValueError("Checkpoint lineage mixes method, seed, or data")
        result.append(run)
        source = meta.get("resume_from_checkpoint")
        if not source:
            return result
        run = Path(source).resolve().parent.parent
    raise ValueError("Cyclic checkpoint lineage")


def choose_keep(entries: list[dict]) -> set[str]:
    latest = sorted(entries, key=lambda e: (e["step"], e["path"]), reverse=True)[:3]
    # Equal scores prefer the earlier checkpoint, preserving a stable best-model choice.
    best = sorted(entries, key=lambda e: (-e["score"], e["step"], e["path"]))[:2]
    return {e["path"] for e in latest + best}


def retain_checkpoints(run: Path, *, apply: bool = False) -> dict:
    run = run.resolve()
    entries, pending = [], []
    for member in lineage(run):
        metrics = read_metrics(member / "metrics.jsonl")
        marker = member / "checkpoints/latest_checkpointed_iteration.txt"
        try:
            saved_through = int(marker.read_text().strip())
        except (FileNotFoundError, ValueError):
            continue
        for cp in (member / "checkpoints").glob("global_step_*"):
            step = checkpoint_step(cp)
            if cp.is_symlink() or step > saved_through or not complete_checkpoint(cp):
                continue
            values = [metrics.get(step, {}).get(k) for k in SCORE_KEYS]
            if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
                pending.append(str(cp))
                continue
            entries.append({"path": str(cp), "step": step, "score": sum(values) / len(values)})
    keep = choose_keep(entries)
    remove = [e for e in entries if e["path"] not in keep]
    # Do not prune while a newly completed checkpoint is awaiting its evaluation.
    if pending:
        remove = []
    plan = {"policy": "union(latest3,best2); no milestone copies", "score_keys": SCORE_KEYS,
            "keep": sorted(keep), "pending_evaluation": pending, "remove": remove,
            "checkpoints": entries, "applied": apply, "time": time.time()}
    if apply:
        journal = run / "retention.jsonl"
        # Record intent before deletion; only complete, scored, unselected checkpoints are removed.
        if remove:
            with journal.open("a") as f:
                f.write(json.dumps(plan, ensure_ascii=False) + "\n")
                f.flush()
            for e in remove:
                cp = Path(e["path"])
                if cp.is_symlink() or cp.resolve().parent.name != "checkpoints" or cp.resolve().parent.parent not in lineage(run):
                    raise ValueError("Refusing checkpoint deletion outside the verified lineage")
                if not complete_checkpoint(cp):
                    raise ValueError("Checkpoint changed during retention; refusing deletion")
                shutil.rmtree(cp)
            for parent in {Path(e['path']).parent for e in remove}:
                remaining = [checkpoint_step(p) for p in parent.glob('global_step_*') if complete_checkpoint(p)]
                marker = parent / 'latest_checkpointed_iteration.txt'
                if remaining:
                    marker.write_text(str(max(remaining)) + '\n')
                elif marker.exists():
                    marker.unlink()
        tmp = run / "retention-state.json.tmp"
        tmp.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n")
        tmp.replace(run / "retention-state.json")
    return plan
