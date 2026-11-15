"""Execute the prepared cross-budget audit on saved reports only; no model imports."""
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def config_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def main():
    out = Path(sys.argv[1]).resolve()
    repo = Path(__file__).resolve().parents[5]
    configs = repo/'configs/runs/segmentation'
    evidence = repo/'docs/handoffs/segmentation/evidence'
    plan = read(configs/'s0_20000_stage_plan.json')
    policy = read(configs/'resume_5000_to_20000.json')
    audit_path = evidence/'s0_20000_preparation_20261115/configuration_check.json'
    audit = read(audit_path)
    git = lambda *args: subprocess.check_output(['git', *args], cwd=repo, text=True).strip()
    head = git('rev-parse', 'HEAD')
    require(not git('status', '--porcelain', '--untracked-files=no'), 'Dirty tracked checkout')
    require(git('rev-parse', 'HEAD:src/fashion_grounding') == policy['target_source_tree']
            == audit['target_source_tree'], 'Unreviewed training/evaluation source tree')
    old_config = read(evidence/'resume5000_20261113/train/resolved_config.json')
    new_config = read(out/'train/resolved_config.json')
    changes = {k: {'before': old_config.get(k), 'after': new_config.get(k)}
               for k in old_config.keys() | new_config.keys()
               if old_config.get(k) != new_config.get(k)}
    require(changes == policy['allowed_changes'], 'Training recipe differs from reviewed transition')
    require(config_sha(old_config) == policy['parent_config_sha256'], 'Old training config mismatch')
    require(config_sha(new_config) == policy['target_config_sha256']
            == audit['target_training_config_sha256'], 'New training config mismatch')
    old_eval = read(evidence/'resume5000_20261113/evaluation/resolved_config.json')
    new_eval = read(out/'dev/resolved_config.json')
    eval_changes = {k: {'before': old_eval.get(k), 'after': new_eval.get(k)}
                    for k in old_eval.keys() | new_eval.keys() if old_eval.get(k) != new_eval.get(k)}
    require(set(eval_changes) == {'budget_seconds', 'run_scope'}, 'AP recipe differs')
    require(config_sha(new_eval) == audit['target_evaluation_config_sha256'], 'New AP config mismatch')
    train_report = read(out/'train/train_report.json')
    require(train_report['status'] == 'completed' and train_report['start_update'] == 5000
            and train_report['completed_updates'] == 20000
            and train_report['updates_this_process'] == 15000, 'Incomplete formal S0')
    require(train_report['all_losses_and_gradients_finite']
            and train_report['skipped_optimizer_updates'] == 0, 'Invalid optimization trajectory')
    old_selection_path = evidence/'resume5000_20261113/selection.json'
    require(sha(old_selection_path) == 'c9bd7e1ba335df7b849b5edacac0c3efd283797585327692a85a4b77382e51cc',
            'Prior selection changed')
    old = read(old_selection_path)
    old_report_path = evidence/'resume5000_20261113/update_003000/evaluation_report.json'
    require(sha(old_report_path) == plan['prior_best_report_sha256'], 'Prior report changed')
    old_report = read(old_report_path)
    require(old['checkpoint_sha256'] == plan['prior_best_checkpoint_sha256']
            == old_report['checkpoint_sha256'], 'Prior checkpoint reference mismatch')
    require(old['completed_updates'] == old_report['completed_updates'] == 3000
            and old['dev_mask_AP'] == old_report['metrics']['AP'] == plan['prior_best_dev_mask_AP'],
            'Prior score/update mismatch')
    new_selection_path = out/'new_budget_selection/selection.json'
    new = read(new_selection_path)
    require(new['compared_updates'] == plan['evaluation_updates'], 'Incomplete evaluation schedule')
    refs = new['evaluation_reports']
    require(len(refs) == len(plan['evaluation_updates']), 'Missing new reports')
    reports = []
    for ref in refs:
        path = Path(ref['path'])
        require(path.parent.parent == out/'dev', 'Report outside this run')
        require(sha(path) == ref['sha256'] == path.with_suffix('.sha256').read_text().strip(),
                'New report hash mismatch')
        report = read(path)
        for key in ('protocol', 'metric_units', 'evaluation_data', 'training_data', 'overlap'):
            require(report[key] == old_report[key], 'Cross-budget evaluation mismatch: '+key)
        require(report['complete'] and report['selection_eligible'] and report['evaluated_images'] == 250
                and report['inference_errors'] == 0 and report['optimizer_updates'] == 0
                and report['checkpoint_model_unchanged'] and not report['queries_read'], 'Ineligible report')
        require(report['training_git_commit'] == report['evaluation_git_commit'] == head
                == train_report['git_commit'], 'Unexpected runtime commit')
        require(report['training_config_sha256'] == config_sha(new_config)
                and report['evaluation_config_sha256'] == config_sha(new_eval), 'Report config mismatch')
        require(math.isfinite(report['metrics']['AP']), 'Non-finite AP')
        reports.append(report)
    require(sorted(r['completed_updates'] for r in reports) == plan['evaluation_updates'], 'Wrong save points')
    best = min(reports, key=lambda r: (-r['metrics']['AP'], r['completed_updates']))
    require(new['completed_updates'] == best['completed_updates']
            and new['checkpoint_sha256'] == best['checkpoint_sha256']
            and new['dev_mask_AP'] == best['metrics']['AP'], 'New selection disagrees with reports')
    selected = min([old, new], key=lambda r: (-r['dev_mask_AP'], r['completed_updates']))
    require(sha(selected['checkpoint_path']) == selected['checkpoint_sha256'], 'Winning weight hash mismatch')
    result = dict(status='completed', rule='max_dev_mask_AP_then_earliest_update',
                  selected={k: selected[k] for k in
                            ('checkpoint_path', 'checkpoint_sha256', 'completed_updates', 'dev_mask_AP')},
                  prior_best=old, new_budget_best=new, old_best_retained=selected is old,
                  new_best_minus_prior_best_AP=new['dev_mask_AP']-old['dev_mask_AP'],
                  compared_updates=[3000]+plan['evaluation_updates'], source_commit=head,
                  source_tree=policy['target_source_tree'], allowed_training_changes=changes,
                  allowed_evaluation_changes=eval_changes,
                  evidence_references=[dict(path=str(p), sha256=sha(p)) for p in
                                       (old_selection_path, old_report_path, new_selection_path, audit_path)],
                  prior_inference_repeated=False, model_imports=0,
                  finished_at=datetime.now(timezone.utc).isoformat())
    with (out/'overall_selection.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result['selected']), flush=True)


if __name__ == '__main__':
    main()
