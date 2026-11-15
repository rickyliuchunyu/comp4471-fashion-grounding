"""Audited eight-point S0 selection across one pinned AMP repair, plus old u3000.

Reports retain their actual identities. The ordinary strict selector is unchanged.
This command reads existing artifacts only; it never imports or runs a model.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from fashion_grounding.data.schema import read_json, write_json, sha256, require
from fashion_grounding.segmentation.evaluation import choose_checkpoint
from fashion_grounding.segmentation.resume_repair import validate_repair
from fashion_grounding.segmentation.training_state import config_sha256, read_config


def select_reports(reports, prior, expected, evaluation_commit, evaluation_config_sha256):
    """Check every identity/denominator before comparing across the approved groups."""
    require(len(reports) == len(expected) == 8, 'Eight complete evaluation points required')
    require(prior['completed_updates'] == 3000, 'Wrong prior best update')
    prior_selected = choose_checkpoint([prior])
    groups, seen = {}, set()
    for report in reports:
        update = report['completed_updates']
        require(update in expected and update not in seen, 'Unexpected or duplicate update')
        seen.add(update)
        for key, value in expected[update].items():
            require(report[key] == value, 'Unexpected checkpoint identity: '+key)
        for key in ('protocol', 'metric_units', 'evaluation_data', 'training_data', 'overlap'):
            require(report[key] == prior[key], 'Incomparable dev protocol/data: '+key)
        require(report['evaluation_git_commit'] == evaluation_commit
                and report['evaluation_config_sha256'] == evaluation_config_sha256, 'Unexpected AP runtime')
        require(report['evaluated_images'] == prior['evaluated_images'] == 250
                and report['optimizer_updates'] == 0 and report['checkpoint_model_unchanged'] is True
                and report['queries_read'] is False, 'Invalid evaluation scope')
        key = (report['training_git_commit'], report['training_config_sha256'])
        groups.setdefault(key, []).append(report)
    # Preserve the strict within-identity gate; the audit above permits only explicit
    # per-checkpoint identities, never a blanket source/config mismatch exemption.
    group_best = [choose_checkpoint(group) for group in groups.values()]
    new_best = min(group_best, key=lambda r: (-r['dev_mask_AP'], r['completed_updates']))
    winner = min([prior_selected, new_best], key=lambda r: (-r['dev_mask_AP'], r['completed_updates']))
    return dict(status='completed', rule='max_dev_mask_AP_then_earliest_update',
                selected=winner, prior_best=prior_selected, new_budget_best=new_best,
                group_selections=group_best, old_best_retained=winner is prior_selected,
                compared_updates=[3000]+sorted(seen),
                new_best_minus_prior_best_AP=new_best['dev_mask_AP']-prior_selected['dev_mask_AP'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--plan', required=True)
    args = p.parse_args()
    out, repo = Path(args.output).resolve(), Path(__file__).resolve().parents[1]
    plan = read_json(args.plan)
    git = lambda *args: subprocess.check_output(['git', *args], cwd=repo, text=True).strip()
    head = git('rev-parse', 'HEAD')
    require(not git('status', '--porcelain', '--untracked-files=no'), 'Dirty source checkout')
    policy_path = repo/plan['repair_policy']
    policy = read_json(policy_path)
    old_config = read_config(repo/plan['parent_config'])
    config = read_config(repo/plan['training_config'])
    eval_config = read_json(repo/plan['evaluation_config'])
    prior_config = read_json(repo/plan['prior_training_config'])
    prior_eval = read_json(repo/plan['prior_evaluation_config'])
    original_transition = read_json(repo/'configs/runs/segmentation/resume_5000_to_20000.json')
    changes = {k: dict(before=prior_config.get(k), after=old_config.get(k))
               for k in prior_config.keys() | old_config.keys() if prior_config.get(k) != old_config.get(k)}
    require(changes == original_transition['allowed_changes'], 'Original budget transition differs')
    require(config_sha256(prior_config) == original_transition['parent_config_sha256'], 'Prior recipe hash differs')
    eval_changes = {k for k in prior_eval.keys() | eval_config.keys() if prior_eval.get(k) != eval_config.get(k)}
    require(eval_changes <= {'budget_seconds', 'run_scope'}, 'AP recipe changed')
    require(config_sha256(eval_config) == plan['evaluation_config_sha256'], 'Unpinned AP config')
    parent = policy['parent_git_commit']
    fake_metadata = dict(config=old_config, config_sha256=config_sha256(old_config),
                         git_commit=parent, completed_updates=18000)
    repair = validate_repair(fake_metadata, config, policy, plan['parent_checkpoint_sha256'], head,
        git('rev-parse', parent+':src/fashion_grounding'), git('rev-parse', 'HEAD:src/fashion_grounding'),
        git('diff', '--name-only', parent, head, '--', 'src/fashion_grounding').splitlines())
    for path in ('model.py', 'evaluate.py', 'evaluation.py', 'predict.py'):
        name = 'src/fashion_grounding/segmentation/'+path
        require(git('rev-parse', original_transition['parent_git_commit']+':'+name)
                == git('rev-parse', 'HEAD:'+name), 'AP implementation changed: '+path)
    train = read_json(out/'train/train_report.json')
    require(train['status'] == 'completed' and train['start_update'] == 18000
            and train['completed_updates'] == 20000 and train['updates_this_process'] == 2000,
            'Recovery training incomplete')
    require(train['git_commit'] == head and train['config_sha256'] == config_sha256(config)
            and train['all_losses_and_gradients_finite'] and train['skipped_optimizer_updates'] == 0,
            'Invalid successful-update trajectory')
    recorded_repair = read_json(out/'train/source_repair.json')
    require(recorded_repair == dict(repair, policy_sha256=sha256(policy_path)), 'Runtime source repair differs')
    require(train['resume_lineage'][-1] == recorded_repair, 'Repair lineage missing')
    resumed = read_json(out/'train/resume_check.json')
    require(resumed['status'] == 'passed' and resumed['resumed_from_update'] == 18000
            and resumed['checkpoint_sha256'] == plan['parent_checkpoint_sha256']
            and resumed['expected_digests'] == resumed['restored_digests']
            and resumed['exact_components'] == ['model', 'optimizer', 'rng', 'sampler', 'scaler', 'scheduler'],
            'Incomplete state restoration')
    prior_path = repo/plan['prior_best_report']
    require(sha256(prior_path) == plan['prior_best_report_sha256'], 'Prior AP report changed')
    prior = read_json(prior_path)
    require(prior['checkpoint_sha256'] == plan['prior_best_checkpoint_sha256']
            and prior['metrics']['AP'] == plan['prior_best_dev_mask_AP'], 'Prior best mismatch')
    expected = {}
    for item in plan['existing_checkpoints']:
        expected[item['update']] = dict(checkpoint_sha256=item['sha256'], checkpoint_path=item['path'],
            training_git_commit=parent, training_config_sha256=config_sha256(old_config))
    require(sorted(expected) == list(range(6000, 18001, 2000)), 'Wrong existing checkpoint list')
    require(len(train['checkpoints']) == 1 and train['checkpoints'][0]['update'] == 20000,
            'Unexpected recovery save point')
    expected[20000] = dict(checkpoint_sha256=train['checkpoints'][0]['sha256'],
        checkpoint_path=str(out/'train/checkpoint_020000.pth'), training_git_commit=head,
        training_config_sha256=config_sha256(config))
    reports, refs = [], []
    for update in sorted(expected):
        path = out/'dev'/f'update_{update:06d}'/'evaluation_report.json'
        require(sha256(path) == path.with_suffix('.sha256').read_text().strip(), 'Report hash mismatch')
        report = read_json(path)
        require(sha256(path.parent/'predictions.json') == report['predictions_sha256'], 'Predictions changed')
        checkpoint = Path(expected[update]['checkpoint_path'])
        require(sha256(checkpoint) == checkpoint.with_suffix('.sha256').read_text().strip()
                == expected[update]['checkpoint_sha256'], 'Checkpoint bytes differ')
        reports.append(report); refs.append(dict(path=str(path), sha256=sha256(path)))
    result = select_reports(reports, prior, expected, head, config_sha256(eval_config))
    require(sha256(result['selected']['checkpoint_path']) == result['selected']['checkpoint_sha256'],
            'Winning weight differs')
    result.update(evaluation_reports=refs, repair=recorded_repair, source_commit=head,
                  prior_best_report_sha256=sha256(prior_path), stage_plan_sha256=sha256(args.plan),
                  prior_inference_repeated=False, model_runs=0,
                  finished_at=datetime.now(timezone.utc).isoformat())
    target = out/'overall_selection.json'
    require(not target.exists(), 'Selection already exists')
    write_json(target, result)
    print(json.dumps(result['selected']), flush=True)


if __name__ == '__main__': main()
