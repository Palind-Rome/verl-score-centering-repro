#!/usr/bin/env python3
"""Resolve and validate all method/mode combinations without starting Ray."""
import argparse
import json
from pathlib import Path
import sys

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from run_experiment import METHODS, REPO, build_overrides


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.upstream))
    from verl.utils.config import validate_config
    from verl.utils import omega_conf_to_dataclass

    report = []
    with initialize_config_dir(version_base=None, config_dir=str(args.upstream / "verl/trainer/config")):
        for stage in ("smoke", "pilot"):
            for mode in ("sync", "separate_async"):
                for method in METHODS:
                    a = argparse.Namespace(root=args.root, upstream=args.upstream, stage=stage,
                        mode=mode, method=method, seed=42, steps=None,
                        model=Path("/mnt/data1/ckpts/chy/models/Qwen/Qwen3-4B"))
                    python = Path(sys.executable)
                    overrides = build_overrides(a, args.root / "runs" / "config-check", Path("/tmp/sc-config-check"), python)
                    cfg = compose(config_name="ppo_trainer", overrides=overrides)
                    OmegaConf.resolve(cfg)
                    validate_config(cfg, use_reference_policy=False, use_critic=False)
                    omega_conf_to_dataclass(cfg.actor_rollout_ref.rollout)
                    report.append({"stage": stage, "mode": mode, "method": method, "valid": True})
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
