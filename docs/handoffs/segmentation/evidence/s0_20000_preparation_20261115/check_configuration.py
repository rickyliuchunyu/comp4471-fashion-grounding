"""Preparation-only metadata audit. No torch import, model/data load or training."""
import ast
from bisect import bisect_right
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
HERE = Path(__file__).resolve().parent
CONFIG = ROOT / 'configs/runs/segmentation'
SRC = ROOT / 'src/fashion_grounding'


def read_json(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def extract(path, names, namespace):
    # Execute only the existing pure config/schedule functions, excluding imports
    # of torch, checkpoint loaders, datasets, evaluators and model code.
    parsed = ast.parse(path.read_text())
    nodes = [node for node in parsed.body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    assert {node.name for node in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)


def main():
    ns = dict(json=json, hashlib=hashlib, Path=Path, np=np,
              bisect_right=bisect_right, read_json=read_json, require=require)
    extract(SRC/'segmentation/training_state.py',
            ['read_config', 'config_sha256', 'checkpoint_due', 'UpdateSchedule'], ns)
    extract(SRC/'segmentation/resume_extension.py', ['validate_extension'], ns)
    ns['BUDGET_KEYS'] = {'max_updates', 'budget_seconds', 'checkpoint_interval',
                         'checkpoint_updates', 'run_scope'}
    extract(SRC/'segmentation/evaluation.py', ['read_eval_config'], ns)
    old = ns['read_config'](CONFIG/'s0_resume_5000.json')
    new = ns['read_config'](CONFIG/'s0_resume_20000.json')
    policy = read_json(CONFIG/'resume_5000_to_20000.json')
    report = read_json(HERE.parent/'resume5000_20261113/train/train_report.json')
    archived_config = read_json(HERE.parent/'resume5000_20261113/train/resolved_config.json')
    assert old == archived_config
    server = read_json(HERE/'server_preflight.json')
    assert server['checkpoint_sha256'] == server['sidecar_sha256'] == policy['parent_checkpoint_sha256']
    assert report['config_sha256'] == policy['parent_config_sha256']
    assert report['git_commit'] == policy['parent_git_commit']
    assert report['completed_updates'] == policy['parent_completed_updates'] == 5000
    assert report['checkpoints'][-1]['sha256'] == server['checkpoint_sha256']
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()
    parent = policy['parent_git_commit']
    target = git('rev-parse', 'HEAD')
    subprocess.run(['git', 'merge-base', '--is-ancestor', parent, target], cwd=ROOT, check=True)
    state = dict(config=old, config_sha256=report['config_sha256'],
                 git_commit=parent, completed_updates=5000)
    extension = ns['validate_extension'](
        state, new, policy, server['checkpoint_sha256'], target,
        git('rev-parse', parent+':src/fashion_grounding'),
        git('rev-parse', target+':src/fashion_grounding'))
    assert set(extension['changes']) == {'budget_seconds', 'max_updates', 'checkpoint_updates', 'run_scope'}
    assert new['max_updates'] - state['completed_updates'] == 15000
    expected_points = list(range(6000, 20001, 2000))
    assert [u for u in range(5001, 20001) if ns['checkpoint_due'](new, u, 20000)] == expected_points
    # Newly reached LR boundaries: inspect the existing schedule without stepping
    # an optimizer or changing RNG. completed=12000 sets LR for update 12001.
    optimizer = SimpleNamespace(param_groups=[{'lr': 1e-4}, {'lr': 1e-5}])
    schedule = ns['UpdateSchedule'](optimizer, new)
    lrs = {}
    for completed, expected in [(5000, 1e-4), (11999, 1e-4), (12000, 1e-5),
                                (15999, 1e-5), (16000, 1e-6), (19999, 1e-6)]:
        schedule.completed_updates = completed
        schedule._apply()
        assert np.isclose(optimizer.param_groups[0]['lr'], expected, rtol=1e-12, atol=0)
        lrs[str(completed+1)] = [g['lr'] for g in optimizer.param_groups]
    old_eval = ns['read_eval_config'](CONFIG/'dev_ap_250_resume_5000.json')
    new_eval = ns['read_eval_config'](CONFIG/'dev_ap_250_resume_20000.json')
    assert {k for k in old_eval.keys() | new_eval.keys() if old_eval.get(k) != new_eval.get(k)} == {'budget_seconds', 'run_scope'}
    plan = read_json(CONFIG/'s0_20000_stage_plan.json')
    assert plan['status'] == 'prepared_not_started' and not plan['execution_authorized_in_current_stage']
    assert plan['evaluation_updates'] == expected_points
    assert plan['start_update'] == 5000 and plan['target_update'] == new['max_updates']
    assert plan['training_internal_budget_seconds'] == new['budget_seconds'] == 8400
    assert plan['evaluation_aggregate_internal_budget_seconds'] == new_eval['budget_seconds'] == 1800
    assert sum(plan[k] for k in ('training_internal_budget_seconds', 'evaluation_aggregate_internal_budget_seconds',
                                'overhead_and_selection_reserve_seconds')) == plan['total_hard_limit_seconds'] == 10800
    assert plan['process_group_kill_after_seconds'] < plan['shared_gpu_wrapper_timeout_seconds']
    assert plan['shared_gpu_wrapper_timeout_seconds'] + plan['wrapper_max_cleanup_seconds'] <= 10800
    # Audit every existing Python file, including transitive train/AP imports.
    changes = git('diff', '--name-status', parent, target, '--', 'src/fashion_grounding').splitlines()
    assert changes and all(line.startswith('A\t') for line in changes)
    existing = git('ls-tree', '-r', '--name-only', parent, 'src/fashion_grounding').splitlines()
    blobs = {}
    for path in existing:
        assert git('rev-parse', parent+':'+path) == git('rev-parse', target+':'+path)
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == hashlib.sha256(
            subprocess.check_output(['git', 'show', parent+':'+path], cwd=ROOT)).hexdigest()
        blobs[path] = git('rev-parse', parent+':'+path)
    baseline_path = HERE.parent/'resume5000_20261113/update_003000/evaluation_report.json'
    baseline_sha = hashlib.sha256(baseline_path.read_bytes()).hexdigest()
    assert baseline_sha == 'ed11d80e093050a21c69983eb6f9f293845ca26f82ec8467e559303632c43163'
    result = dict(status='passed', scope='new S0 configuration and pinned transition metadata only',
                  source_snapshot=target, target_source_tree=policy['target_source_tree'],
                  target_training_config_sha256=ns['config_sha256'](new),
                  target_evaluation_config_sha256=ns['config_sha256'](new_eval),
                  stage_plan_sha256=hashlib.sha256((CONFIG/'s0_20000_stage_plan.json').read_bytes()).hexdigest(),
                  transition=extension, checkpoint_updates=expected_points,
                  learning_rates_by_next_update=lrs, unchanged_parent_source_blobs=blobs,
                  added_source_files=changes, prior_best_report_sha256=baseline_sha,
                  model_imports=0, checkpoint_deserializations=0, gpu_calls=0,
                  optimizer_updates=0, old_smoke_runs=0,
                  note='Parent report metadata reused; actual 5000-to-20000 state restoration is not executed.')
    (HERE/'configuration_check.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('status', 'target_training_config_sha256',
                                          'target_evaluation_config_sha256', 'checkpoint_updates')}))


if __name__ == '__main__':
    main()
