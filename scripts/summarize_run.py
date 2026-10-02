#!/usr/bin/env python3
"""Inspect local run evidence without mistaking process completion for learning."""
import argparse
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run = args.run_dir
    launch = json.loads((run / "launch.json").read_text())
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines() if line]
    training = [r for r in rows if "actor/pg_loss" in r["data"]]
    keys = ["actor/pg_loss", "actor/grad_norm", "actor/entropy_loss", "actor/ppo_kl",
            "actor/sc_correction", "actor/sc_sampler_head_mass", "actor/sc_train_head_mass",
            "critic/rewards/mean", "critic/rewards/min", "critic/rewards/max",
            "critic/advantages/mean", "response_length/mean", "response_length/clip_ratio",
            "timing_s/step", "timing_s/testing"]
    result = {"run_id": launch["run_id"], "stage": launch["stage"],
              "method": launch["method"], "mode": launch["mode"],
              "training_steps_logged": [r["step"] for r in training],
              "exit": json.loads((run / "exit.json").read_text()) if (run / "exit.json").exists() else None,
              "metrics": {}, "nonfinite_or_null": []}
    for key in keys:
        items = [(r["step"], r["data"][key]) for r in rows if key in r["data"]]
        if items:
            result["metrics"][key] = {"first": items[0], "last": items[-1]}
    for row in training:
        for key, value in row["data"].items():
            if value is None or isinstance(value, (int, float)) and not math.isfinite(value):
                result["nonfinite_or_null"].append([row["step"], key])
    if launch["method"] in ("sc", "sc_tis"):
        result["sc_diagnostics_present_every_training_step"] = bool(training) and all(
            all(key in row["data"] for key in
                ("actor/sc_correction", "actor/sc_sampler_head_mass", "actor/sc_train_head_mass"))
            for row in training)
    gradients = [r["data"].get("actor/grad_norm", 0) for r in training]
    result["nonzero_gradient_observed"] = any(isinstance(v, (int, float)) and v > 0 for v in gradients)
    result["interpretation"] = "Smoke completion checks execution only; it does not establish a benchmark benefit."
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
