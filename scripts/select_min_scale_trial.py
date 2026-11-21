"""Offline B20k vs saved A20k, retaining the old best unless overall AP improves."""
import argparse
import json
from pathlib import Path
import subprocess

from fashion_grounding.data.schema import read_json, write_json, sha256, require
from fashion_grounding.segmentation.evaluation import choose_checkpoint
from fashion_grounding.segmentation.training_state import config_sha256, read_config
from run_min_scale_trial import validate_plan, ROOT


def compare_reports(control, trial, best):
    """Use strict single-report eligibility, then permit the audited recipe identities."""
    for report in (control, trial, best):
        choose_checkpoint([report])
        require(report['evaluated_images'] == 250 and report['optimizer_updates'] == 0
                and report['checkpoint_model_unchanged'] is True and report['queries_read'] is False,
                'Invalid full-dev scope')
        for key in ('protocol', 'metric_units', 'evaluation_config_sha256', 'evaluation_data',
                    'training_data', 'overlap'):
            require(report[key] == control[key], 'Incomparable AP/data: '+key)
    require(control['completed_updates'] == trial['completed_updates'] == 20000
            and best['completed_updates'] == 18000, 'Wrong comparison updates')
    # Stable ties retain a published result; the historical 18k point wins update ties.
    choices = [(best, 'prior_best_18000'), (control, 'control_A_20000'), (trial, 'trial_B_20000')]
    winner, group = min(choices, key=lambda item: (-item[0]['metrics']['AP'], item[0]['completed_updates']))
    delta = {k: trial['metrics'][k]-control['metrics'][k]
             for k in ('AP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large')}
    delta['per_category_AP'] = {k: trial['metrics']['per_category_AP'][k]-v
                              for k, v in control['metrics']['per_category_AP'].items()}
    return dict(status='completed', selected_group=group, selected=choose_checkpoint([winner]),
        B_minus_A_AP_points=delta, B_minus_prior_best_AP=trial['metrics']['AP']-best['metrics']['AP'],
        control_inference_repeated=False, model_runs=0, scientific_factor={'min_scale': [0.1, 0.5]},
        interpretation='One low-LR seed42 continuation screen; no statistical significance claim')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    out, plan = Path(args.output).resolve(), read_json(args.plan)
    audit = validate_plan(plan)
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    train = read_json(out/'train/train_report.json')
    config = read_config(ROOT/plan['training_config'])
    require(train['status'] == 'completed' and train['start_update'] == 18000
            and train['completed_updates'] == 20000 and train['updates_this_process'] == 2000
            and train['git_commit'] == head and train['config_sha256'] == config_sha256(config)
            and train['all_losses_and_gradients_finite'] and train['skipped_optimizer_updates'] == 0,
            'Incomplete or different B training')
    require(read_json(out/'train/recipe_transition.json') == audit
            and train['resume_lineage'][-1] == audit, 'Missing recipe transition lineage')
    resumed = read_json(out/'train/resume_check.json')
    require(resumed['status'] == 'passed' and resumed['resumed_from_update'] == 18000
            and resumed['checkpoint_sha256'] == plan['parent_checkpoint_sha256']
            and resumed['expected_digests'] == resumed['restored_digests']
            and resumed['exact_components'] == ['model', 'optimizer', 'rng', 'sampler', 'scaler', 'scheduler'],
            'Incomplete six-state restoration')
    require(len(train['checkpoints']) == 1 and train['checkpoints'][0]['update'] == 20000,
            'Unexpected B checkpoint list')
    checkpoint = out/'train/checkpoint_020000.pth'
    digest = sha256(checkpoint)
    require(digest == checkpoint.with_suffix('.sha256').read_text().strip()
            == train['checkpoints'][0]['sha256'], 'B checkpoint bytes changed')
    report_path = out/'dev/update_020000/evaluation_report.json'
    require(sha256(report_path) == report_path.with_suffix('.sha256').read_text().strip(), 'B report changed')
    trial = read_json(report_path)
    require(trial['checkpoint_sha256'] == digest and trial['checkpoint_path'] == str(checkpoint)
            and trial['training_git_commit'] == trial['evaluation_git_commit'] == head
            and trial['training_config_sha256'] == config_sha256(config)
            and trial['predictions_sha256'] == sha256(report_path.parent/'predictions.json'), 'Wrong B identity')
    control, best = (read_json(ROOT/plan[k]) for k in ('control_report', 'prior_best_report'))
    result = compare_reports(control, trial, best)
    require(sha256(result['selected']['checkpoint_path']) == result['selected']['checkpoint_sha256'],
            'Selected checkpoint bytes changed')
    result.update(recipe_transition=audit, stage_plan_sha256=sha256(args.plan),
        evidence={k: dict(path=plan[k], sha256=sha256(ROOT/plan[k]))
                  for k in ('control_report', 'prior_best_report')},
        trial_report_sha256=sha256(report_path))
    require(not (out/'comparison.json').exists(), 'Comparison already exists')
    write_json(out/'comparison.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
