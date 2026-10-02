#!/usr/bin/env python3
"""Launch an isolated, version-pinned verl SC experiment. No shared Ray cleanup."""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import uuid

REPO = Path(__file__).resolve().parents[1]
UPSTREAM_SHA = json.loads((REPO / "upstream.json").read_text())["commit"]
METHODS = {"pg": (False, None), "tis": (False, "token"),
           "sc": (True, None), "sc_tis": (True, "token")}


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=METHODS, default="sc_tis")
    p.add_argument("--mode", choices=["sync", "separate_async"], default="sync")
    p.add_argument("--stage", choices=["smoke", "pilot", "main"], default="smoke")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int)
    p.add_argument("--gpus", help="Explicit comma-separated local GPU indices")
    p.add_argument("--root", type=Path, default=Path("/mnt/data2/Palind/score-centering-repro"))
    p.add_argument("--upstream", type=Path, default=REPO.parent / "verl")
    p.add_argument("--python", type=Path)
    p.add_argument("--model", type=Path,
                   default=Path("/mnt/data1/ckpts/chy/models/Qwen/Qwen3-4B"))
    p.add_argument("--swanlab-mode", choices=["offline", "cloud"], default="offline")
    p.add_argument("--dry-run", action="store_true", help="Print command, do not access GPUs or create a run")
    p.add_argument("--config-only", action="store_true", help="Resolve Hydra config without launching Ray")
    return p.parse_args()


def build_overrides(a: argparse.Namespace, run_dir: Path, ray_tmp: Path, python: Path) -> list[str]:
    sc, importance = METHODS[a.method]
    smoke = a.stage == "smoke"
    separate = a.mode == "separate_async"
    batch, n, response = (8, 4, 512) if smoke else (32, 5, 8192)
    trainer_gpus = 4 if separate else (2 if smoke else 8)
    rollout_gpus = 4 if separate else trainer_gpus
    data = a.root / "datasets" / "prepared"
    values = {
        "trainer.use_v1": True,
        "trainer.v1.trainer_mode": a.mode,
        "trainer.v1.separate_async.parameter_sync_step": 1,
        "trainer.v1.separate_async.num_warmup_batches": 1,
        "trainer.v1.separate_async.hybrid_rollout.enable_switch": False,
        "trainer.v1.sampler.max_off_policy_threshold": 4,
        "trainer.v1.sampler.max_off_policy_strategy": "drop",
        "transfer_queue.enable": True,
        "actor_rollout_ref.hybrid_engine": not separate,
        "data.train_files": str(data / ("smoke_train.parquet" if smoke else "train.parquet")),
        "data.val_files": ([str(data / "smoke_val.parquet")] if smoke else
                           [str(data / "aime24.parquet"), str(data / "aime25.parquet")]),
        "data.train_batch_size": batch,
        "data.gen_batch_size": 1,
        "data.max_prompt_length": 1024,
        "data.max_response_length": response,
        "data.filter_overlong_prompts": True,
        "data.truncation": "error",
        "data.return_raw_chat": True,
        "data.shuffle": True,
        "data.seed": a.seed,
        "data.dataloader_num_workers": 4,
        "+data.apply_chat_template_kwargs.enable_thinking": not smoke,
        "algorithm.adv_estimator": "grpo",
        "algorithm.norm_adv_by_std_in_grpo": False,
        "algorithm.filter_groups.enable": False,
        "algorithm.use_kl_in_reward": False,
        "algorithm.rollout_correction.bypass_mode": True,
        "algorithm.rollout_correction.loss_type": "reinforce",
        "algorithm.rollout_correction.rollout_is": importance,
        "algorithm.rollout_correction.rollout_is_threshold": 2.0,
        "algorithm.rollout_correction.rollout_rs": None,
        "algorithm.rollout_correction.rollout_is_batch_normalize": False,
        "algorithm.rollout_correction.score_centering": sc,
        "actor_rollout_ref.model.path": str(a.model),
        "actor_rollout_ref.model.use_remove_padding": True,
        "actor_rollout_ref.model.use_fused_kernels": False,
        "actor_rollout_ref.model.enable_gradient_checkpointing": True,
        "actor_rollout_ref.actor.strategy": "fsdp2",
        "actor_rollout_ref.actor.fsdp_config.fsdp_size": -1,
        "actor_rollout_ref.actor.fsdp_config.seed": a.seed,
        "actor_rollout_ref.actor.fsdp_config.param_offload": False,
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload": False,
        "actor_rollout_ref.actor.fsdp_config.dtype": "bfloat16",
        "actor_rollout_ref.actor.optim.lr": 1e-6,
        "actor_rollout_ref.actor.optim.lr_warmup_steps": -1,
        "actor_rollout_ref.actor.optim.lr_warmup_steps_ratio": 0.0,
        "actor_rollout_ref.actor.optim.weight_decay": 0.01,
        "actor_rollout_ref.actor.grad_clip": 1.0,
        "actor_rollout_ref.actor.ppo_mini_batch_size": batch,
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.actor.ppo_epochs": 1,
        "actor_rollout_ref.actor.use_dynamic_bsz": True,
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": 4096 if smoke else 12288,
        "actor_rollout_ref.actor.use_fused_kernels": False,
        "actor_rollout_ref.actor.use_torch_compile": not smoke,
        "actor_rollout_ref.actor.fsdp_config.use_torch_compile": not smoke,
        "actor_rollout_ref.actor.loss_agg_mode": "token-mean",
        "actor_rollout_ref.actor.policy_loss.loss_mode": "bypass_mode",
        "actor_rollout_ref.actor.data_loader_seed": a.seed,
        "actor_rollout_ref.actor.use_kl_loss": False,
        "actor_rollout_ref.actor.entropy_coeff": 0.0,
        "actor_rollout_ref.actor.calculate_entropy": True,
        "actor_rollout_ref.actor.entropy_from_logits_with_chunking": True,
        "actor_rollout_ref.rollout.name": "vllm",
        "actor_rollout_ref.rollout.mode": "async",
        "actor_rollout_ref.rollout.nnodes": 1 if separate else 0,
        "actor_rollout_ref.rollout.n_gpus_per_node": rollout_gpus,
        "actor_rollout_ref.rollout.tensor_model_parallel_size": 2,
        "actor_rollout_ref.rollout.gpu_memory_utilization": 0.6 if separate else 0.4,
        "actor_rollout_ref.rollout.dtype": "bfloat16",
        "actor_rollout_ref.rollout.n": n,
        "actor_rollout_ref.rollout.seed": a.seed,
        "actor_rollout_ref.rollout.temperature": 1.0,
        "actor_rollout_ref.rollout.top_p": 1.0,
        "actor_rollout_ref.rollout.top_k": -1,
        "actor_rollout_ref.rollout.calculate_log_probs": True,
        "actor_rollout_ref.rollout.topk_log_probs": 128 if sc else 0,
        "actor_rollout_ref.rollout.logprobs_mode": "processed_logprobs",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz": True,
        "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu": 4096 if smoke else 12288,
        "actor_rollout_ref.rollout.max_model_len": 1024 + response,
        "actor_rollout_ref.rollout.max_num_seqs": 32,
        "actor_rollout_ref.rollout.max_num_batched_tokens": 4096 if smoke else 12288,
        "actor_rollout_ref.rollout.enforce_eager": smoke,
        "actor_rollout_ref.rollout.checkpoint_engine.backend": "nccl",
        "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes": 1024,
        "actor_rollout_ref.rollout.val_kwargs.n": 1 if smoke else 8,
        "actor_rollout_ref.rollout.val_kwargs.do_sample": True,
        "actor_rollout_ref.rollout.val_kwargs.temperature": 1.0,
        "actor_rollout_ref.rollout.val_kwargs.top_p": 1.0,
        "actor_rollout_ref.rollout.val_kwargs.top_k": -1,
        "reward.custom_reward_function.path": None,
        "reward.custom_reward_function.name": "compute_score",
        "reward.reward_manager.name": "naive",
        "reward.num_workers": 4,
        "trainer.logger": ["console", "file", "swanlab"],
        "trainer.project_name": "verl-score-centering-repro",
        "trainer.experiment_name": run_dir.name,
        "trainer.nnodes": 1,
        "trainer.n_gpus_per_node": trainer_gpus,
        "trainer.total_epochs": 100,
        "trainer.total_training_steps": a.steps or {"smoke": 2, "pilot": 20, "main": 200}[a.stage],
        "trainer.val_before_train": not smoke,
        "trainer.test_freq": 2 if smoke else 20,
        "trainer.save_freq": -1 if smoke else 20,
        "trainer.max_actor_ckpt_to_keep": 3,
        "trainer.resume_mode": "disable",
        "trainer.default_local_dir": str(run_dir / "checkpoints"),
        "trainer.validation_data_dir": str(run_dir / "validation"),
        "trainer.rollout_data_dir": str(run_dir / "rollouts"),
        "trainer.log_val_generations": 4,
        "ray_kwargs.ray_init.num_cpus": 32,
        "+ray_kwargs.ray_init.address": "local",
        "+ray_kwargs.ray_init.include_dashboard": False,
        "+ray_kwargs.ray_init.namespace": "sc-" + run_dir.name,
        "+ray_kwargs.ray_init._temp_dir": str(ray_tmp),
        "ray_kwargs.ray_init.runtime_env.py_executable": str(python),
        "hydra.run.dir": str(run_dir / "hydra"),
    }
    # The actor mirror is not in the base Hydra schema; explicitly add every field.
    for k in ("bypass_mode", "loss_type", "rollout_is", "rollout_is_threshold",
              "rollout_rs", "rollout_is_batch_normalize", "score_centering"):
        values[f"+actor_rollout_ref.actor.policy_loss.rollout_correction.{k}"] = values[
            f"algorithm.rollout_correction.{k}"]
    return [f"{k}={json.dumps(v, ensure_ascii=False, separators=(',', ':'))}" for k, v in values.items()]


def preflight(a: argparse.Namespace, gpu_ids: list[int]) -> dict:
    actual_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=a.upstream, text=True).strip()
    if actual_sha != UPSTREAM_SHA:
        raise RuntimeError(f"Expected upstream {UPSTREAM_SHA}, got {actual_sha}")
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                               cwd=a.upstream, text=True).strip():
        raise RuntimeError("Upstream tracked files are modified; record/review changes before running")
    if not (a.model / "model.safetensors.index.json").is_file():
        raise RuntimeError("Expected existing sharded base model")
    index = json.loads((a.model / "model.safetensors.index.json").read_text())
    if not all((a.model / p).is_file() for p in set(index["weight_map"].values())):
        raise RuntimeError("Base model shard missing")
    prepared = a.root / "datasets" / "prepared"
    manifest = json.loads((prepared / "manifest.json").read_text())
    current_reward = hashlib.sha256((a.upstream / "verl/utils/reward_score/math_dapo.py").read_bytes()).hexdigest()
    if manifest["conversion"]["reward_sha256"] != current_reward:
        raise RuntimeError("Reward code changed since dataset preparation; regenerate the manifest")
    expected_sources = json.loads((REPO / "data" / "sources.lock.json").read_text())["sources"]
    if manifest["sources"] != expected_sources:
        raise RuntimeError("Dataset manifest differs from source lock")
    for name, spec in manifest["outputs"].items():
        if hashlib.sha256((prepared / name).read_bytes()).hexdigest() != spec["sha256"]:
            raise RuntimeError(f"Prepared dataset checksum mismatch: {name}")
    listing = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"
    ], text=True)
    available = {}
    for line in listing.strip().splitlines():
        idx, uid, mem, util = (x.strip() for x in line.split(","))
        available[int(idx)] = {"uuid": uid, "memory_mib": int(mem), "utilization": int(util)}
    processes = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"
    ], text=True)
    for idx in gpu_ids:
        gpu = available[idx]
        if gpu["uuid"] in processes or gpu["memory_mib"] > 100 or gpu["utilization"] > 0:
            raise RuntimeError(f"GPU {idx} is occupied; refusing to start")
    return {"upstream_commit": actual_sha, "gpus": {i: available[i] for i in gpu_ids}}


def main() -> int:
    a = arguments()
    a.root = a.root.resolve()
    a.upstream = a.upstream.resolve()
    python = a.python or a.root / "envs" / "verl" / "bin" / "python"
    run_id = f"{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}-{a.stage}-{a.mode}-{a.method}-s{a.seed}-{uuid.uuid4().hex[:6]}"
    run_dir = a.root / "runs" / run_id
    # A short symlink keeps Unix socket paths below Linux's 108-byte limit.
    ray_tmp = Path("/tmp") / ("scr-" + uuid.uuid4().hex[:8])
    gpu_ids = [int(x) for x in (a.gpus or ("0,1" if a.stage == "smoke" and a.mode == "sync"
                                          else "0,1,2,3,4,5,6,7")).split(",")]
    expected = 2 if a.stage == "smoke" and a.mode == "sync" else 8
    if len(gpu_ids) != expected or len(set(gpu_ids)) != expected:
        raise ValueError(f"This configuration needs {expected} distinct GPUs")
    command = [str(python), "-m", "verl.trainer.main_ppo", *build_overrides(a, run_dir, ray_tmp, python)]
    if a.config_only:
        command += ["--cfg", "job", "--resolve"]
    if a.dry_run:
        print(shlex.join(command))
        return 0
    if a.stage != "smoke" and a.swanlab_mode != "cloud" and not a.config_only:
        raise RuntimeError("Pilot/main require SwanLab cloud logging; complete project-local login first")
    env = dict(os.environ)
    env.update({
        "CUDA_VISIBLE_DEVICES": ",".join(map(str, gpu_ids)),
        "PYTHONPATH": os.pathsep.join([str(REPO), str(a.upstream)]),
        "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "4",
        "RAY_ADDRESS": "local", "RAY_USAGE_STATS_ENABLED": "0",
        "RAY_ENABLE_UV_RUN_RUNTIME_ENV": "0",
        "HF_HOME": str(a.root / "cache" / "huggingface"),
        "TORCH_HOME": str(a.root / "cache" / "torch"),
        "TORCH_EXTENSIONS_DIR": str(a.root / "cache" / "torch_extensions"),
        "TRITON_CACHE_DIR": str(a.root / "cache" / "triton"),
        "CUDA_CACHE_PATH": str(a.root / "cache" / "cuda"),
        "XDG_CACHE_HOME": str(a.root / "cache" / "xdg"),
        "TMPDIR": str(a.root / "tmp"),
        "SWANLAB_MODE": a.swanlab_mode,
        "SWANLAB_LOG_DIR": str(run_dir / "swanlog"),
        "VERL_FILE_LOGGER_PATH": str(run_dir / "metrics.jsonl"),
    })
    cuda_home = python.parent.parent / "lib/python3.12/site-packages/nvidia/cu13"
    if (cuda_home / "bin/nvcc").is_file():
        env["CUDA_HOME"] = str(cuda_home)
        env["PATH"] = str(cuda_home / "bin") + os.pathsep + env.get("PATH", "")
    if a.config_only:
        return subprocess.call(command, cwd=REPO, env=env)
    a.root.mkdir(parents=True, exist_ok=True)
    with (a.root / "runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        hardware = preflight(a, gpu_ids)
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "ray").mkdir()
        ray_tmp.symlink_to(run_dir / "ray", target_is_directory=True)
        dataset_manifest = a.root / "datasets" / "prepared" / "manifest.json"
        manifest_sha = hashlib.sha256(dataset_manifest.read_bytes()).hexdigest()
        record = {"run_id": run_id, "stage": a.stage, "method": a.method, "mode": a.mode,
                  "seed": a.seed, "command": command, "hardware": hardware,
                  "dataset_manifest_sha256": manifest_sha, "ray_tmp": str(ray_tmp),
                  "note": "Smoke uses shorter responses and thinking disabled; not a benchmark result."}
        if (REPO / ".git").exists():
            record["reproduction_commit"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
            record["reproduction_dirty"] = bool(subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=REPO, text=True).strip())
        (run_dir / "launch.json").write_text(json.dumps(record, indent=2) + "\n")
        print(f"RUN_DIR={run_dir}", flush=True)
        with (run_dir / "console.log").open("w", buffering=1) as log:
            child = subprocess.Popen(command, cwd=REPO, env=env, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            (run_dir / "driver.pid").write_text(str(child.pid) + "\n")
            code = child.wait()
        (run_dir / "exit.json").write_text(json.dumps({"returncode": code,
            "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat()}) + "\n")
        print(f"EXIT_CODE={code}", flush=True)
        return code


if __name__ == "__main__":
    sys.exit(main())
