# Score Centering in verl: Qwen3-4B reproduction

This repository prepares a controlled comparison of PG, token-level TIS,
Score Centering (SC), and SC + TIS using the merged verl implementation.
It is a reproduction workspace, not a report of successful training.

The upstream code and dataset revisions are pinned in `upstream.json`.
The initial target is single-turn mathematical reasoning with Qwen3-4B,
BF16 FSDP2 training, and vLLM rollout. The asynchronous configuration uses
four training GPUs and four rollout GPUs. This extends the upstream PR's
small-model experiment; it is not an exact replication of the paper's JAX
experiments or its FP8 hardware configuration.

Current validation: 37 reward-parser tests pass; data conversion yields 17,398
training rows and 30 rows in each AIME evaluation set. Environment and GPU
end-to-end checks are tracked separately; these counts are not training results.

## Environment

Use a new, dedicated environment. Existing shared environments and the base
model are read-only inputs. Place the environment, caches, datasets, and
checkpoints on a data disk.

```bash
export SC_ROOT=/path/to/data/score-centering-repro
export UV_PROJECT_ENVIRONMENT="$SC_ROOT/envs/verl"
export UV_CACHE_DIR="$SC_ROOT/cache/uv"
export TMPDIR="$SC_ROOT/tmp"
mkdir -p "$TMPDIR" "$UV_CACHE_DIR"
git clone https://github.com/verl-project/verl.git ../verl
git -C ../verl checkout 8718ca30a3f002f93b7c4fd99b9b2506718681bc
(cd ../verl && uv sync --frozen --extra fsdp --extra vllm --extra math --python 3.12)
uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" -r requirements-extra.txt
```

Run the resulting interpreter directly so that subsequent uv synchronization
does not remove the additional tracking package. Save the full installed
package list with each environment build.

## Data and reward

The public source is `open-r1/DAPO-Math-17k-Processed`, configuration `all`,
split `train`. The preparation script preserves the original `source_prompt`
instructions and records source revisions, file hashes, duplicates, conflicting
labels, and train/evaluation overlap. AIME24 and AIME25 are evaluation inputs,
not training inputs. See `scripts/prepare_data.py --help` for preparation options.

The reward function extracts an explicitly marked final integer answer and
returns a signed correctness reward. Its parsing rules and failure diagnostics
are covered by tests. It is independently defined here because the previous
experiment's private reward implementation is unavailable.

## Checks and execution

```bash
PY="$SC_ROOT/envs/verl/bin/python"
"$PY" scripts/validate_config.py --upstream ../verl --root "$SC_ROOT"
"$PY" -m pytest tests/test_reward.py
"$PY" scripts/run_experiment.py --stage smoke --mode sync --method sc_tis \
  --root "$SC_ROOT" --upstream ../verl --model /path/to/Qwen3-4B
```

The smoke configuration has two steps, 512 response tokens, and thinking
disabled. It validates execution and is not a benchmark result. Smoke logs are
saved with SwanLab in offline mode. A subsequent asynchronous smoke run uses
`--mode separate_async` and the eight-GPU split.
Its eight diagnostic validation questions are deliberately drawn from the smoke
training subset; they must not be reported as held-out benchmark accuracy.

Pilot and main configurations use thinking, 8192 response tokens, 32 prompts,
five responses per prompt, one optimizer pass per batch, AdamW at 1e-6, and
token-mean loss. TIS uses threshold 2. All four arms use the same data and
hyperparameters except the SC/TIS switches. SC arms collect top-128 logprobs;
their additional rollout overhead may affect realized policy age, which must
be reported when interpreting the asynchronous comparison.

For cloud tracking, log in locally from this repository directory:

```bash
"$SC_ROOT/envs/verl/bin/swanlab" login --local
"$PY" scripts/run_experiment.py --stage pilot --mode separate_async \
  --method sc_tis --swanlab-mode cloud --root "$SC_ROOT" \
  --upstream ../verl --model /path/to/Qwen3-4B
```

Use `--method pg`, `tis`, `sc`, or `sc_tis`. `--dry-run` prints the command;
`--config-only` resolves the Hydra configuration without starting Ray.

Every run has a new output directory containing launch metadata, full console
logs, `metrics.jsonl`, generated validation/rollout text, and checkpoints when
enabled. The launcher checks the upstream revision and GPU occupancy, takes a
local experiment lock, and requests a new Ray instance explicitly. It never
uses global `ray stop`, kills unrelated jobs, or writes to the base model.

## Interpretation

Do not infer that a working smoke run validates SC's claimed benefit. Inspect
held-out accuracy, signed reward, entropy, response truncation, gradient norm,
SC head mass/correction, policy staleness, and importance-weight diagnostics.
Measure throughput before scheduling long runs. Use multiple seeds before
making a statistical claim. Published dataset labels can contain errors;
preparation audit reports must accompany results.

## References

- [verl PR #8010](https://github.com/verl-project/verl/pull/8010)
- [Score Centering paper](https://arxiv.org/html/2609.20807v1)
- [Author implementation](https://github.com/martin-marek/score-centering)
- [Processed DAPO dataset](https://huggingface.co/datasets/open-r1/DAPO-Math-17k-Processed)

The reproduction tooling was prepared with AI assistance and must be evaluated
by its recorded checks and experiment results.
