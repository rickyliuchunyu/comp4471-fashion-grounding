"""Receipt and selection contracts for the segmented 10k adaptation run.

No model loads, inference, or data re-audit. Existing strict within-run selection
remains unchanged; the cached baseline participates in a separate comparison.
"""
import json
from pathlib import Path

from ..data.schema import require, sha256
from .evaluation import choose_checkpoint


def read(path):
    return json.loads(Path(path).read_text())


def write_atomic(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def select_progress(baseline, reports, epochs):
    """Recompute history, including plateau count, from complete dev receipts."""
    base = choose_checkpoint([baseline])
    require(bool(reports) and len(reports) == len(epochs), 'Missing evaluation stages')
    require(epochs == [2, 4, 6, 8, 10][:len(epochs)], 'Nonconsecutive evaluation stages')
    new_best = choose_checkpoint(reports)  # Never add the 5k report to this list.
    common = ('protocol', 'metric_units', 'evaluation_config_sha256', 'evaluation_data', 'evaluated_images')
    incumbent, stale = base['dev_mask_AP'], 0
    for epoch, report in zip(epochs, reports):
        require(all(report[k] == baseline[k] for k in common), 'Baseline dev protocol/population differs')
        require(report['completed_updates'] == epoch * 5000, 'Wrong epoch/update evaluation')
        score = report['metrics']['AP']
        stale = 0 if score > incumbent else stale + 1
        incumbent = max(incumbent, score)
    use_new = new_best['dev_mask_AP'] > base['dev_mask_AP']
    chosen = next(r for r in reports if r['checkpoint_sha256'] == new_best['checkpoint_sha256']) if use_new else baseline
    reason = ('plateau' if stale >= 3 and epochs[-1] >= 4 else
              'max_equivalent_epochs' if epochs[-1] == 10 else 'continue')
    return dict(status='complete_dev_selection', selected_origin='tenk' if use_new else 'cached_C5k20k',
        selected_checkpoint=new_best if use_new else base, new_run_best=new_best,
        cached_baseline=base, completed_equivalent_epochs=epochs[-1],
        consecutive_without_new_best=stale, stop_reason=reason,
        cross_run_rule='max_full_dev_mask_AP; keep cached baseline on tie',
        selected_minus_baseline={k: chosen['metrics'][k] - baseline['metrics'][k]
                                for k in ('AP', 'AP_small', 'AP75')},
        per_category_AP=chosen['metrics'].get('per_category_AP'),
        further_diagnostics='Reuse saved predictions for small-target matches and boundary analysis; not computed by this selector')


def audit_training(directory, stage, previous, context):
    directory = Path(directory)
    report = read(directory/'train_report.json')
    update = stage['new_run_update']
    start = previous['completed_updates'] if previous else 0
    require(report['status'] == 'completed' and report['purpose'] == 's0_training'
            and report['git_commit'] == context['git_commit']
            and report['config_sha256'] == context['training_config_sha256'], 'Training identity differs')
    require(report['start_update'] == start and report['completed_updates'] == update
            and report['updates_this_process'] == update-start
            and report['sample_count'] == 2*(update-start)
            and report['resumed'] is bool(previous), 'Training exposure/resume differs')
    require(report['all_losses_and_gradients_finite'] is True
            and report['skipped_optimizer_updates'] == 0
            and report['backbone_and_classifier_updated'] is True, 'Training did not finish valid updates')
    data = read(directory/'data_check.json')
    require(all(data[k] == v for k, v in context['training_data'].items())
            and data['images'] == 10000 and data['instances'] == 39699,
            'Training data receipt differs')
    initialization = read(directory/'initialization.json')
    require(initialization['mode'] == 'model_only' and initialization['source'] == context['model_only_source']
            and initialization['new_completed_updates'] == 0
            and initialization['loaded_components'] == ['model']
            and set(initialization['reset_components']) == {'optimizer', 'scheduler', 'scaler', 'sampler', 'rng', 'amp_recovery'}
            and initialization['classifier_reinitialized'] is False, 'Wrong model-only initialization')
    if previous:
        resume = read(directory/'resume_check.json')
        components = {'model', 'optimizer', 'scheduler', 'scaler', 'sampler', 'rng'}
        require(resume['status'] == 'passed' and resume['resumed_from_update'] == start
                and resume['checkpoint_sha256'] == previous['checkpoint_sha256']
                and set(resume['exact_components']) == components
                and set(resume['expected_digests']) == components
                and resume['expected_digests'] == resume['restored_digests'], 'Resume state differs')
    checkpoints = [c for c in report['checkpoints'] if c['update'] == update]
    require(len(checkpoints) == 1, 'Missing or duplicate terminal checkpoint')
    checkpoint = checkpoints[0]
    path = directory/f'checkpoint_{update:06d}.pth'
    require(checkpoint['path'] == path.name and path.is_file()
            and path.with_suffix('.sha256').read_text().strip() == checkpoint['sha256'],
            'Checkpoint receipt/sidecar differs')
    # The evaluator checks the full checkpoint SHA and model fingerprints. Do not
    # deserialize the real checkpoint again merely to repeat that validation.
    return dict(completed_updates=update, checkpoint_path=str(path), checkpoint_sha256=checkpoint['sha256'])


def audit_evaluation(directory, checkpoint, context):
    directory = Path(directory)
    run = read(directory/'run_report.json')
    path = directory/f"update_{checkpoint['completed_updates']:06d}"/'evaluation_report.json'
    require(run['status'] == 'completed' and run['reports'] == [str(path)]
            and run['optimizer_updates'] == 0 and run['purpose'] == 'dev_checkpoint_selection',
            'Full dev stage incomplete')
    require(path.with_suffix('.sha256').read_text().strip() == sha256(path), 'Dev report digest differs')
    report = read(path)
    choose_checkpoint([report])
    require(all(report[k] == v for k, v in checkpoint.items())
            and report['training_git_commit'] == report['evaluation_git_commit'] == context['git_commit']
            and report['training_config_sha256'] == context['training_config_sha256']
            and report['evaluation_config_sha256'] == context['evaluation_config_sha256']
            and report['evaluation_data'] == context['evaluation_data']
            and report['training_data'] == context['training_data'], 'Dev/checkpoint identity differs')
    require(report['evaluated_images'] == 250 and report['optimizer_updates'] == 0
            and report['checkpoint_model_unchanged'] is True and report['queries_read'] is False,
            'Wrong full dev population or evaluation behavior')
    require(set(report['overlap']) == {'image_id', 'sha256', 'group_id', 'source_identity'}
            and all(v == 0 for v in report['overlap'].values()), 'Wrong partition isolation receipt')
    require(sha256(path.parent/'predictions.json') == report['predictions_sha256'], 'New prediction digest differs')
    return report
