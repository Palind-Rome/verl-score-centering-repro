import json
from pathlib import Path

import pytest

from sc_repro.checkpoints import SCORE_KEYS, choose_keep, complete_checkpoint, retain_checkpoints, lineage
from scripts.run_matrix import inspect_metrics


def make_run(root, name='run', parent=None, method='sc_tis'):
    r = root/'runs'/name
    r.mkdir(parents=True)
    (r/'launch.json').write_text(json.dumps({'method':method,'seed':42,'dataset_manifest_sha256':'same',
                                            'resume_from_checkpoint':str(parent) if parent else None}))
    return r


def checkpoint(run, step, score=None):
    p = run/'checkpoints'/f'global_step_{step}'
    names = [f'actor/{kind}_world_size_4_rank_{rank}.pt'
             for kind in ['model','optim','extra_state'] for rank in range(4)]
    names += ['data.pt','transfer_queue/controller_state.pkl','transfer_queue/metadata.json',
              'transfer_queue/simple_storage/storage_unit_info.json']
    names += [f'transfer_queue/simple_storage/su_{i}.pkl' for i in range(8)]
    for name in names:
        f=p/name;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(b'x')
    (p.parent/'latest_checkpointed_iteration.txt').write_text(str(step))
    if score is not None:
        with (run/'metrics.jsonl').open('a') as f:
            f.write(json.dumps({'step':step,'data':{k:score for k in SCORE_KEYS}})+'\n')
    return p


def test_best_inside_latest_has_no_extra_copy():
    entries=[{'path':str(s),'step':s,'score':s/1000} for s in [20,40,60,80,100]]
    assert choose_keep(entries)=={'60','80','100'}


def test_healthy_old_best_survive_three_collapsed_recent_checkpoints():
    entries=[{'path':str(s),'step':s,'score':v} for s,v in [(20,.5),(40,.6),(60,0),(80,0),(100,0)]]
    assert choose_keep(entries)=={'20','40','60','80','100'}


def test_union_prunes_only_unselected_checkpoint_no_copy(tmp_path):
    r=make_run(tmp_path)
    paths=[checkpoint(r,s,v) for s,v in [(20,.7),(40,.6),(60,.1),(80,.2),(100,.3),(120,.4)]]
    plan=retain_checkpoints(r,apply=True)
    assert {Path(p).name for p in plan['keep']}=={'global_step_20','global_step_40','global_step_80','global_step_100','global_step_120'}
    assert not paths[2].exists()
    assert len(list((r/'checkpoints').glob('global_step_*')))==5
    assert all(complete_checkpoint(p) for p in paths if p.exists())


def test_checkpoint_awaiting_evaluation_blocks_pruning(tmp_path):
    r=make_run(tmp_path)
    for s in [20,40,60,80,100]:checkpoint(r,s,.5)
    p=checkpoint(r,120)
    plan=retain_checkpoints(r,apply=True)
    assert str(p) in plan['pending_evaluation'] and plan['remove']==[]
    assert len(list((r/'checkpoints').glob('global_step_*')))==6


def test_lineage_retention_deduplicates_source_and_new_run(tmp_path):
    old=make_run(tmp_path,'old')
    for s,v in [(160,.4),(180,.6),(200,.5)]:source=checkpoint(old,s,v)
    new=make_run(tmp_path,'new',source)
    for s in [220,240,260]:checkpoint(new,s,.3)
    plan=retain_checkpoints(new,apply=True)
    assert len(plan['keep'])==5
    assert not (old/'checkpoints/global_step_160').exists()
    assert source.exists()
    assert (old/'checkpoints/global_step_180').exists()


def test_cross_method_lineage_is_refused(tmp_path):
    old=make_run(tmp_path,'old',method='pg');source=checkpoint(old,200,.5)
    new=make_run(tmp_path,'new',source)
    with pytest.raises(ValueError,match='mixes'):lineage(new)


def test_missing_optimizer_is_not_complete(tmp_path):
    r=make_run(tmp_path);p=checkpoint(r,20,.5)
    (p/'actor/optim_world_size_4_rank_3.pt').unlink()
    assert not complete_checkpoint(p)


def test_unpublished_checkpoint_is_not_a_resume_source(tmp_path):
    r=make_run(tmp_path);p=checkpoint(r,20,.5)
    (p.parent/'latest_checkpointed_iteration.txt').unlink()
    assert not complete_checkpoint(p)


def test_continuation_requires_201_through_500_not_500_new_steps(tmp_path):
    data={'actor/pg_loss':0.,'actor/grad_norm':.1}
    (tmp_path/'metrics.jsonl').write_text(''.join(json.dumps({'step':s,'data':data})+'\n' for s in range(201,501)))
    assert inspect_metrics(tmp_path,'pg',500,finished=True,start_step=200)['training_steps']==300
    with pytest.raises(RuntimeError):inspect_metrics(tmp_path,'pg',500,finished=True)
