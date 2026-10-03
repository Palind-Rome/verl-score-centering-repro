#!/usr/bin/env python3
"""Persistently run the four SC comparison arms in sequence; never clean up shared Ray.

Run under nohup/systemd to survive SSH disconnects. --first-run adopts an existing
first arm. --resume-matrix resumes monitoring saved state without retrying failed
arms or relaunching an arm whose launch outcome is uncertain.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

REPO = Path(__file__).resolve().parents[1]
LAUNCHER = REPO / "scripts/run_experiment.py"
DEFAULT_ROOT = Path("/mnt/data2/Palind/score-centering-repro")
METHODS = ("sc_tis", "sc", "tis", "pg")
ACTOR_REQUIRED = ("actor/pg_loss", "actor/grad_norm")
SC_REQUIRED = ("actor/sc_correction", "actor/sc_sampler_head_mass", "actor/sc_train_head_mass")
ACTOR_OPTIONAL = ("actor/entropy_loss", "actor/ppo_kl", "actor/total_loss", "actor/loss")
LOG = logging.getLogger("sc_matrix")


def utcnow():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command_overrides(command: list[str]) -> dict:
    result = {}
    for token in command:
        if "=" not in token:
            continue
        key, raw = token.split("=", 1)
        try:
            result[key.lstrip("+")] = json.loads(raw)
        except json.JSONDecodeError:
            result[key.lstrip("+")] = raw
    return result


def validate_run(run_dir: Path, method: str, seed: int, steps: int, model: Path | None = None) -> dict:
    launch = json.loads((run_dir / "launch.json").read_text())
    for key, expected in {"stage": "main", "mode": "separate_async", "method": method, "seed": seed}.items():
        if launch.get(key) != expected:
            raise RuntimeError(f"{run_dir}: {key} must be {expected!r}, got {launch.get(key)!r}")
    overrides = command_overrides(launch.get("command", []))
    source = launch.get("resume_from_checkpoint")
    start_step = launch.get("start_step", 0)
    if source:
        if Path(source).parent.parent.parent != run_dir.parent or not Path(source).name.startswith("global_step_"):
            raise RuntimeError("Resume checkpoint must belong to the same runs root")
        if start_step != int(Path(source).name.split("global_step_")[-1]) or not 0 < start_step < steps:
            raise RuntimeError("Invalid continuation start step")
        parent = json.loads((Path(source).parent.parent / "launch.json").read_text())
        if parent['method'] != method or parent['seed'] != seed or parent['dataset_manifest_sha256'] != launch['dataset_manifest_sha256']:
            raise RuntimeError("Continuation lineage differs in method, seed or data")
        if overrides.get('trainer.resume_from_path') != source:
            raise RuntimeError("Resume path differs from launch metadata")
    elif start_step != 0:
        raise RuntimeError("Nonzero start step requires a checkpoint")
    for key, expected in {"trainer.total_training_steps": steps, "trainer.resume_mode": "resume_path" if source else "disable",
                          "trainer.v1.trainer_mode": "separate_async"}.items():
        if overrides.get(key) != expected:
            raise RuntimeError(f"{run_dir}: {key} must be {expected!r}, got {overrides.get(key)!r}")
    if model is not None and Path(overrides.get("actor_rollout_ref.model.path", "")).resolve() != model.resolve():
        raise RuntimeError(f"{run_dir}: base model path differs from the matrix")
    checkpoint = overrides.get("trainer.default_local_dir")
    if checkpoint is None or Path(checkpoint).resolve() != (run_dir / "checkpoints").resolve():
        raise RuntimeError(f"{run_dir}: command does not target this exact run directory")
    expected_sc = method in ("sc", "sc_tis")
    expected_is = "token" if method in ("tis", "sc_tis") else None
    if overrides.get("algorithm.rollout_correction.score_centering") is not expected_sc:
        raise RuntimeError(f"{run_dir}: score centering setting differs from method {method}")
    if overrides.get("algorithm.rollout_correction.rollout_is") != expected_is:
        raise RuntimeError(f"{run_dir}: importance sampling setting differs from method {method}")
    return launch


def inspect_metrics(run_dir: Path, method: str, steps: int, *, finished: bool = False, start_step: int = 0) -> dict:
    """Inspect complete JSONL records; tolerate only an unfinished final write."""
    path = run_dir / "metrics.jsonl"
    raw = path.read_bytes() if path.exists() else b""
    lines = raw.splitlines(keepends=True)
    if lines and not lines[-1].endswith(b"\n"):
        if finished:
            raise RuntimeError("Metrics end in an incomplete JSONL record")
        lines.pop()
    training_steps = set()
    diagnostic_nonfinite = []
    required = ACTOR_REQUIRED + (SC_REQUIRED if method in ("sc", "sc_tis") else ())
    for line_number, line in enumerate(lines, 1):
        try:
            row = json.loads(line)
            data = row["data"]
        except (ValueError, KeyError, TypeError) as exc:
            raise RuntimeError(f"Malformed metric record at line {line_number}: {exc}") from exc
        if not isinstance(data, dict):
            raise RuntimeError(f"Metric data is not an object at line {line_number}")
        if not any(key.startswith("actor/") for key in data):
            continue
        step = row.get("step")
        if isinstance(step, bool) or not isinstance(step, int) or not start_step < step <= steps:
            raise RuntimeError(f"Unexpected training step {step!r}; expected {start_step+1}..{steps}")
        core = required + tuple(key for key in ACTOR_OPTIONAL if key in data)
        for key in core:
            value = data.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise RuntimeError(f"Nonfinite, null, missing, or nonnumeric {key} at step {step}: {value!r}")
        # orjson converts NaN/Inf into null. Non-core diagnostics such as IS
        # moments can overflow even while the optimizer remains finite: record
        # them for analysis, but do not alter the experiment's stopping rule.
        for key, value in data.items():
            if key not in core and (value is None or isinstance(value, (int, float)) and not math.isfinite(value)):
                diagnostic_nonfinite.append({"step": step, "key": key, "value": repr(value)})
        training_steps.add(step)
    expected = set(range(start_step + 1, steps + 1))
    if finished and training_steps != expected:
        missing = sorted(expected - training_steps)
        raise RuntimeError(f"Expected all {steps} training steps; observed {len(training_steps)}; missing {missing[:12]}")
    return {"training_steps": len(training_steps), "last_step": max(training_steps, default=0),
            "required_actor_metrics": list(required), "diagnostic_nonfinite": diagnostic_nonfinite}


def verify_completion(run_dir: Path, method: str, seed: int, steps: int, model: Path | None = None) -> dict:
    launch = validate_run(run_dir, method, seed, steps, model)
    result = json.loads((run_dir / "exit.json").read_text())
    if result.get("returncode") != 0:
        raise RuntimeError(f"Run exited unsuccessfully: {result.get('returncode')!r}")
    return inspect_metrics(run_dir, method, steps, finished=True, start_step=launch.get('start_step', 0))


def driver_identity(run_dir: Path) -> tuple[int, str] | None:
    """Verify exact run identity before observing or signaling any process."""
    pid_file = run_dir / "driver.pid"
    if not pid_file.exists():
        return None
    pid = int(pid_file.read_text().strip())
    proc = Path("/proc") / str(pid)
    try:
        argv = [arg.decode(errors="replace") for arg in (proc / "cmdline").read_bytes().split(b"\0") if arg]
        stat = (proc / "stat").read_text().rsplit(")", 1)[1].split()
        if not argv or stat[0] == "Z":
            return None
        if proc.stat().st_uid != os.getuid() or os.getpgid(pid) != pid:
            raise RuntimeError(f"Refusing driver PID {pid}: wrong owner or process-group leader")
        overrides = command_overrides(argv)
        if "verl.trainer.main_ppo" not in argv or overrides.get("trainer.default_local_dir") != str(run_dir / "checkpoints"):
            raise RuntimeError(f"Refusing driver PID {pid}: command does not identify this exact run")
        return pid, stat[19]  # /proc stat field 22, process start ticks
    except (FileNotFoundError, ProcessLookupError):
        return None


def terminate_verified_driver(run_dir: Path) -> str:
    identity = driver_identity(run_dir)
    if identity is None:
        return "Driver is already absent; no signal sent"
    # Verify again immediately before signaling to narrow the PID-reuse race.
    if driver_identity(run_dir) != identity:
        raise RuntimeError("Driver identity changed; no signal sent")
    try:
        os.killpg(identity[0], signal.SIGTERM)
    except ProcessLookupError:
        return "Driver exited before SIGTERM"
    return f"Sent SIGTERM only to verified driver process group {identity[0]}"


def pid_belongs_to_run(pid: int, run_dir: Path | None) -> bool:
    if run_dir is None:
        return False
    try:
        fields = (Path("/proc") / str(pid) / "environ").read_bytes().split(b"\0")
        expected = {f"VERL_FILE_LOGGER_PATH={run_dir / 'metrics.jsonl'}".encode(),
                    f"SWANLAB_LOG_DIR={run_dir / 'swanlog'}".encode()}
        return any(field in expected for field in fields)
    except (FileNotFoundError, PermissionError):
        return False


def wait_for_free_gpus(previous_run: Path | None, timeout: int, poll: int):
    """Only query GPUs. Unknown GPU processes stop the queue immediately."""
    deadline = time.monotonic() + timeout
    while True:
        query = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
                                         "--format=csv,noheader,nounits"], text=True, timeout=20)
        gpus = {}
        for line in query.splitlines():
            idx, uid, memory, utilization = [field.strip() for field in line.split(",")]
            if int(idx) in range(8):
                gpus[uid] = (int(memory), int(utilization))
        if len(gpus) != 8:
            raise RuntimeError("Matrix requires all eight expected GPUs")
        processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                                             "--format=csv,noheader,nounits"], text=True, timeout=20)
        active = []
        for line in processes.splitlines():
            if not line.strip():
                continue
            uid, pid_text = [field.strip() for field in line.split(",")]
            if uid not in gpus:
                continue
            pid = int(pid_text)
            if not pid_belongs_to_run(pid, previous_run):
                if not (Path("/proc") / str(pid)).exists():
                    continue  # Process exited between nvidia-smi and identity inspection.
                raise RuntimeError(f"GPU occupied by another or unidentifiable process PID {pid}; stopping queue")
            active.append(pid)
        idle = not active and all(memory <= 100 and utilization == 0 for memory, utilization in gpus.values())
        if idle:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError("GPUs did not become idle within the release timeout; no process was killed")
        time.sleep(min(poll, max(0.1, deadline - time.monotonic())))


def build_command(a, method: str) -> list[str]:
    command = [str(a.python), str(LAUNCHER), "--stage", "main", "--mode", "separate_async",
            "--method", method, "--steps", str(a.steps), "--seed", str(a.seed), "--swanlab-mode", "cloud",
            "--root", str(a.root), "--upstream", str(a.upstream), "--python", str(a.python), "--model", str(a.model)]
    source = getattr(a, 'resume_checkpoints', {}).get(method)
    if source:
        command += ['--resume-from-checkpoint', str(source)]
    return command


def planned_entries(a) -> list[dict]:
    return [{"method": method, "status": "adopted" if i == 0 and a.first_run else "pending",
             "run_dir": str(a.first_run) if i == 0 and a.first_run else None,
             "command": build_command(a, method)} for i, method in enumerate(a.methods)]


def save_state(matrix_dir: Path, state: dict):
    state["updated_utc"] = utcnow()
    temporary = matrix_dir / "state.json.tmp"
    with temporary.open("w") as stream:
        json.dump(state, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(matrix_dir / "state.json")


def find_launched_run(log_path: Path, root: Path) -> Path | None:
    if not log_path.exists():
        return None
    for line in log_path.read_text(errors="replace").splitlines():
        if line.startswith("RUN_DIR="):
            candidate = Path(line[len("RUN_DIR="):]).resolve()
            if candidate.parent != (root / "runs").resolve():
                raise RuntimeError("Launcher returned a run path outside the matrix root")
            return candidate
    return None


def assert_launcher_unchanged(state):
    if digest(LAUNCHER) != state["launcher_sha256"]:
        raise RuntimeError("run_experiment.py changed during the matrix; queue stopped to preserve comparability")


def run_queue(a, matrix_dir: Path, state: dict):
    previous_run = None
    for index, entry in enumerate(state["runs"]):
        assert_launcher_unchanged(state)
        if entry["status"] == "complete":
            previous_run = Path(entry["run_dir"])
            verify_completion(previous_run, entry["method"], a.seed, a.steps, a.model)
            continue
        if entry["status"] == "failed":
            raise RuntimeError("A failed arm is retained in state; this queue does not automatically retry or skip it")
        state["active_index"] = index
        child = None
        log_path = matrix_dir / f"launcher-{index}-{entry['method']}.log"
        if entry["status"] == "pending":
            wait_for_free_gpus(previous_run, a.gpu_release_timeout, a.poll_seconds)
            assert_launcher_unchanged(state)
            entry.update(status="launching", launch_started_utc=utcnow(), launcher_log=str(log_path))
            save_state(matrix_dir, state)  # Persist uncertain launch before spawning: never auto-double-launch.
            with log_path.open("a", buffering=1) as stream:
                child = subprocess.Popen(entry["command"], cwd=REPO, stdout=stream,
                                         stderr=subprocess.STDOUT, start_new_session=True)
            entry["launcher_pid"] = child.pid
            save_state(matrix_dir, state)
            LOG.info("Launching %s; launcher PID %s", entry["method"], child.pid)
        if not entry.get("run_dir"):
            deadline = time.monotonic() + 120
            while True:
                run_dir = find_launched_run(log_path, a.root)
                if run_dir is not None:
                    entry["run_dir"] = str(run_dir)
                    break
                if child is not None and child.poll() is not None:
                    raise RuntimeError(f"Launcher exited before reporting a run: {child.returncode}; see {log_path}")
                if time.monotonic() >= deadline:
                    raise RuntimeError("Launch outcome is uncertain; refusing to launch a duplicate arm")
                time.sleep(a.poll_seconds)
        run_dir = Path(entry["run_dir"]).resolve()
        launch = validate_run(run_dir, entry["method"], a.seed, a.steps, a.model)
        entry.update(status="running", run_dir=str(run_dir), display_name=launch.get("display_name"))
        save_state(matrix_dir, state)
        LOG.info("Monitoring %s at %s", entry["method"], run_dir)
        absent_since = None
        last_reported_step = None
        while True:
            assert_launcher_unchanged(state)
            try:
                progress = inspect_metrics(run_dir, entry["method"], a.steps, start_step=launch.get('start_step',0))
            except RuntimeError as error:
                try:
                    action = terminate_verified_driver(run_dir)
                except Exception as signal_error:
                    action = f"No signal sent: {signal_error}"
                entry.update(status="failed", failure=str(error), termination=action)
                save_state(matrix_dir, state)
                raise RuntimeError(f"Numerical/metric failure: {error}. {action}") from error
            if progress["last_step"] != last_reported_step:
                entry["progress"] = progress
                save_state(matrix_dir, state)
                LOG.info("%s: %s/%s steps", entry["method"], progress["last_step"], a.steps)
                last_reported_step = progress["last_step"]
            if (run_dir / "exit.json").exists():
                summary = verify_completion(run_dir, entry["method"], a.seed, a.steps, a.model)
                if child is not None:
                    child.wait(timeout=60)
                    if child.returncode != 0:
                        raise RuntimeError(f"Launcher returned nonzero {child.returncode} despite run exit.json")
                entry.update(status="complete", completed_utc=utcnow(), progress=summary)
                save_state(matrix_dir, state)
                LOG.info("Completed %s (%s steps)", entry["method"], a.steps)
                previous_run = run_dir
                break
            identity = driver_identity(run_dir)
            if identity is None:
                absent_since = absent_since or time.monotonic()
                if time.monotonic() - absent_since >= 60:
                    raise RuntimeError("Driver disappeared without exit.json; queue stopped")
            else:
                absent_since = None
                entry["driver_pid"] = identity[0]
            if child is not None and child.poll() is not None:
                raise RuntimeError(f"Launcher exited unexpectedly with code {child.returncode} and no exit.json")
            time.sleep(a.poll_seconds)
    state.update(status="complete", completed_utc=utcnow())
    save_state(matrix_dir, state)
    LOG.info("All matrix arms completed")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--upstream", type=Path, default=REPO.parent / "verl")
    parser.add_argument("--python", type=Path)
    parser.add_argument("--model", type=Path, default=Path("/mnt/data1/ckpts/chy/models/Qwen/Qwen3-4B"))
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--resume-checkpoint", action='append', default=[], metavar='METHOD=PATH')
    parser.add_argument("--first-run", type=Path)
    parser.add_argument("--resume-matrix", type=Path)
    parser.add_argument("--poll-seconds", type=int, default=20, choices=range(15, 31))
    parser.add_argument("--gpu-release-timeout", type=int, default=120, choices=range(60, 121))
    parser.add_argument("--dry-run", action="store_true")
    a = parser.parse_args()
    a.root = a.root.resolve()
    a.upstream = a.upstream.resolve()
    a.model = a.model.resolve()
    a.python = (a.python or a.root / "envs/verl/bin/python").absolute()
    a.methods = a.methods.split(",")
    a.resume_checkpoints = {}
    for item in a.resume_checkpoint:
        method, source = item.split('=', 1)
        if method not in METHODS or method in a.resume_checkpoints:
            parser.error('Invalid or duplicate resume-checkpoint method')
        a.resume_checkpoints[method] = Path(source).resolve()
    if not a.methods or len(set(a.methods)) != len(a.methods) or any(x not in METHODS for x in a.methods):
        parser.error("methods must be a nonempty, nonrepeating subset of sc_tis,sc,tis,pg")
    if a.steps < 1:
        parser.error("steps must be positive")
    if a.first_run:
        a.first_run = a.first_run.resolve()
        if a.first_run.parent != a.root / "runs":
            parser.error("first-run must be inside ROOT/runs")
    if a.resume_matrix and a.first_run:
        parser.error("resume-matrix and first-run are mutually exclusive")
    return a


def main():
    a = parse_args()
    state = None
    if a.resume_matrix:
        matrix_dir = a.resume_matrix.resolve()
        state = json.loads((matrix_dir / "state.json").read_text())
        stored = state["settings"]
        for key in ("root", "upstream", "python", "model"):
            setattr(a, key, Path(stored[key]))
        for key in ("seed", "steps", "poll_seconds", "gpu_release_timeout"):
            setattr(a, key, stored[key])
        a.methods = [entry["method"] for entry in state["runs"]]
        if matrix_dir.parent != a.root / "matrices":
            raise RuntimeError("Saved matrix directory does not belong to its recorded root")
    if a.dry_run:
        entries = state["runs"] if state else planned_entries(a)
        print(json.dumps({"action": "plan_only", "runs": entries}, indent=2))
        return 0
    a.root.mkdir(parents=True, exist_ok=True)
    with (a.root / "matrix.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if state is None:
            name = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
            matrix_dir = a.root / "matrices" / name
            matrix_dir.mkdir(parents=True)
            state = {"schema_version": 1, "created_utc": utcnow(), "status": "running",
                     "launcher_sha256": digest(LAUNCHER), "launcher_path": str(LAUNCHER),
                     "settings": {key: str(getattr(a, key)) for key in ("root", "upstream", "python", "model")},
                     "runs": planned_entries(a)}
            state["settings"].update({key: getattr(a, key) for key in ("seed", "steps", "poll_seconds", "gpu_release_timeout")})
        LOG.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        for handler in (logging.FileHandler(matrix_dir / "controller.log"), logging.StreamHandler()):
            handler.setFormatter(formatter)
            LOG.addHandler(handler)
        (matrix_dir / "controller.pid").write_text(str(os.getpid()) + "\n")
        state.update(controller_pid=os.getpid(), status="running")
        save_state(matrix_dir, state)
        print(f"MATRIX_DIR={matrix_dir}", flush=True)
        try:
            run_queue(a, matrix_dir, state)
        except BaseException as error:
            state.update(status="stopped", stopped_utc=utcnow(), reason=str(error))
            active = state.get("active_index")
            if active is not None and state["runs"][active]["status"] not in ("complete", "failed"):
                state["runs"][active]["last_controller_error"] = str(error)
            save_state(matrix_dir, state)
            LOG.exception("Matrix stopped: %s", error)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
