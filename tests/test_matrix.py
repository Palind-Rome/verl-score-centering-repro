"""CPU-only regression checks for queue adoption and safety conditions."""
import argparse
import json
from pathlib import Path

import pytest

from scripts import run_matrix as matrix


def settings(tmp_path, methods=None, first=None):
    return argparse.Namespace(root=tmp_path, upstream=tmp_path / "verl", python=Path("/usr/bin/python3"),
                              model=tmp_path / "base", steps=200, seed=42, methods=methods or ["sc_tis"],
                              first_run=first, gpu_release_timeout=120, poll_seconds=20)


def write_run(tmp_path, method="sc_tis", stage="main", planned_steps=200, logged_steps=200):
    run = tmp_path / "runs" / "existing"
    run.mkdir(parents=True)
    command = ["python", "-m", "verl.trainer.main_ppo",
               f"trainer.total_training_steps={planned_steps}", 'trainer.resume_mode="disable"',
               'trainer.v1.trainer_mode="separate_async"',
               f'trainer.default_local_dir={json.dumps(str(run / "checkpoints"))}',
               f'actor_rollout_ref.model.path={json.dumps(str(tmp_path / "base"))}',
               "algorithm.rollout_correction.score_centering=" + json.dumps(method in ("sc", "sc_tis")),
               "algorithm.rollout_correction.rollout_is=" + json.dumps("token" if method in ("tis", "sc_tis") else None)]
    (run / "launch.json").write_text(json.dumps({"stage": stage, "mode": "separate_async", "method": method,
                                               "seed": 42, "command": command}))
    (run / "exit.json").write_text(json.dumps({"returncode": 0}))
    data = {"actor/pg_loss": 0.0, "actor/grad_norm": 0.0, "actor/sc_correction": 0.0,
            "actor/sc_sampler_head_mass": 0.5, "actor/sc_train_head_mass": 0.5}
    rows = [{"step": step, "data": dict(data)} for step in range(1, logged_steps + 1)]
    (run / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return run


def test_completed_first_run_is_adopted_without_launching_again(tmp_path, monkeypatch):
    run = write_run(tmp_path)
    a = settings(tmp_path, first=run)
    entries = matrix.planned_entries(a)
    assert entries[0]["status"] == "adopted"
    state = {"runs": entries, "launcher_sha256": matrix.digest(matrix.LAUNCHER)}
    directory = tmp_path / "matrix"
    directory.mkdir()
    def forbidden(*args, **kwargs):
        pytest.fail("An adopted completed run must not launch a process or query GPUs")
    monkeypatch.setattr(matrix.subprocess, "Popen", forbidden)
    monkeypatch.setattr(matrix, "wait_for_free_gpus", forbidden)
    matrix.run_queue(a, directory, state)
    assert state["status"] == "complete"
    assert state["runs"][0]["run_dir"] == str(run)
    assert state["runs"][0]["progress"]["training_steps"] == 200


@pytest.mark.parametrize("stage,planned_steps,logged_steps", [("smoke", 2, 2), ("main", 200, 2)])
def test_two_step_smoke_or_short_run_is_not_a_completed_200_step_arm(tmp_path, stage, planned_steps, logged_steps):
    run = write_run(tmp_path, stage=stage, planned_steps=planned_steps, logged_steps=logged_steps)
    with pytest.raises(RuntimeError):
        matrix.verify_completion(run, "sc_tis", 42, 200)


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), -float("inf")])
def test_nonfinite_or_orjson_null_core_metric_is_rejected(tmp_path, bad):
    run = write_run(tmp_path, logged_steps=1)
    row = json.loads((run / "metrics.jsonl").read_text())
    row["data"]["actor/grad_norm"] = bad
    (run / "metrics.jsonl").write_text(json.dumps(row) + "\n")
    with pytest.raises(RuntimeError, match="actor/grad_norm"):
        matrix.inspect_metrics(run, "sc_tis", 200)


def test_nonfinite_is_diagnostic_is_recorded_without_stopping_finite_actor(tmp_path):
    run = write_run(tmp_path, logged_steps=1)
    row = json.loads((run / "metrics.jsonl").read_text())
    row["data"]["rollout_corr/chi2"] = None
    (run / "metrics.jsonl").write_text(json.dumps(row) + "\n")
    result = matrix.inspect_metrics(run, "sc_tis", 200)
    assert result["last_step"] == 1
    assert result["diagnostic_nonfinite"] == [{"step": 1, "key": "rollout_corr/chi2", "value": "None"}]


def test_finite_zero_gradient_and_loss_are_valid_experiment_results(tmp_path):
    run = write_run(tmp_path)
    assert matrix.verify_completion(run, "sc_tis", 42, 200)["training_steps"] == 200


def test_absent_sc_diagnostic_rejects_sc_but_not_pg(tmp_path):
    run = write_run(tmp_path, logged_steps=1)
    row = json.loads((run / "metrics.jsonl").read_text())
    del row["data"]["actor/sc_correction"]
    (run / "metrics.jsonl").write_text(json.dumps(row) + "\n")
    with pytest.raises(RuntimeError, match="actor/sc_correction"):
        matrix.inspect_metrics(run, "sc", 200)
    assert matrix.inspect_metrics(run, "pg", 200)["last_step"] == 1


def test_changed_launcher_hash_stops_queue():
    with pytest.raises(RuntimeError, match="changed"):
        matrix.assert_launcher_unchanged({"launcher_sha256": "incorrect"})


def test_no_signal_sent_when_driver_identity_changes(tmp_path, monkeypatch):
    identities = iter([(123, "start1"), (123, "start2")])
    monkeypatch.setattr(matrix, "driver_identity", lambda run: next(identities))
    monkeypatch.setattr(matrix.os, "killpg", lambda *args: pytest.fail("Must not signal changed process identity"))
    with pytest.raises(RuntimeError, match="identity changed"):
        matrix.terminate_verified_driver(tmp_path)


def test_all_commands_start_main_from_base_with_expected_shared_settings(tmp_path):
    a = settings(tmp_path, list(matrix.METHODS))
    entries = matrix.planned_entries(a)
    assert [entry["method"] for entry in entries] == ["sc_tis", "sc", "tis", "pg"]
    for entry in entries:
        command = entry["command"]
        assert command[0] == str(a.python)
        assert command[command.index("--stage") + 1] == "main"
        assert command[command.index("--steps") + 1] == "200"
        assert command[command.index("--seed") + 1] == "42"
        assert command[command.index("--model") + 1] == str(a.model)
        assert "--first-run" not in command


def test_resumed_arm_restores_matching_parent_and_counts_only_remaining_steps(tmp_path):
    parent = write_run(tmp_path)
    old = json.loads((parent/'launch.json').read_text())
    old['dataset_manifest_sha256']='same'
    (parent/'launch.json').write_text(json.dumps(old))
    run = tmp_path/'runs'/'continued';run.mkdir()
    source = parent/'checkpoints/global_step_200'
    command = [x for x in old['command'] if not x.startswith(('trainer.total_training_steps=', 'trainer.resume_mode=', 'trainer.default_local_dir='))]
    command += ['trainer.total_training_steps=500','trainer.resume_mode="resume_path"',
                'trainer.resume_from_path='+json.dumps(str(source)),
                'trainer.default_local_dir='+json.dumps(str(run/'checkpoints'))]
    new={**old,'command':command,'start_step':200,'resume_from_checkpoint':str(source)}
    (run/'launch.json').write_text(json.dumps(new));(run/'exit.json').write_text('{"returncode":0}')
    data=json.loads((parent/'metrics.jsonl').read_text().splitlines()[0])['data']
    (run/'metrics.jsonl').write_text(''.join(json.dumps({'step':s,'data':data})+'\n' for s in range(201,501)))
    assert matrix.verify_completion(run,'sc_tis',42,500)['training_steps']==300
    new['method']='sc';(run/'launch.json').write_text(json.dumps(new))
    with pytest.raises(RuntimeError,match='lineage'):
        matrix.validate_run(run,'sc',42,500)


def test_resume_sources_are_assigned_only_to_their_own_methods(tmp_path):
    a=settings(tmp_path,list(matrix.METHODS));a.steps=500
    a.resume_checkpoints={'sc_tis':tmp_path/'global_step_200','sc':tmp_path/'global_step_20'}
    entries=matrix.planned_entries(a)
    for e in entries:
        c=e['command']
        assert c[c.index('--steps')+1]=='500'
        if e['method'] in a.resume_checkpoints:
            assert c[c.index('--resume-from-checkpoint')+1]==str(a.resume_checkpoints[e['method']])
        else:assert '--resume-from-checkpoint' not in c
