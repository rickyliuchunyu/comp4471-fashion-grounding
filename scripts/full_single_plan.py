"""Resolve the single full-release run using the cached 50k report and pinned sampling order."""
from pathlib import Path

from prepare_tenk_adaptation import stage_commands
from fashion_grounding.data.schema import require, sha256
from fashion_grounding.segmentation.evaluation import choose_checkpoint, read_eval_config
from fashion_grounding.segmentation.input_receipts import read_evidence_bundle, validate_chain
from fashion_grounding.segmentation.tenk_execution import read
from fashion_grounding.segmentation.training_state import read_config, config_sha256


def prepare_full(plan, root, authorized):
    combo = plan['profile'] == 'full_combo'
    required = {plan[k] for k in ('training_config', 'model_only_source', 'input_validation_bundle',
                                  'evaluation_config', 'baseline_report', 'sampling_receipt')}
    require(required == set(plan['input_sha256'])
            and all(sha256(root/p) == h for p, h in plan['input_sha256'].items()), 'Full plan input changed')
    config = read_config(root/plan['training_config'])
    spec = config['sampling']
    receipt = read(root/plan['sampling_receipt'])
    bundle, refs, _ = read_evidence_bundle(root/plan['input_validation_bundle'])
    validate_chain(bundle, refs)
    baseline = read(root/plan['baseline_report'])
    source = read(root/plan['model_only_source'])
    evaluation = read_eval_config(root/plan['evaluation_config'])
    choose_checkpoint([baseline])
    updates = plan['evaluation_updates']
    require(config['training_authorized'] is False and config['initialization_mode'] == 'model_only'
            and config['input_bundle_sha256'] == plan['input_sha256'][plan['input_validation_bundle']]
            and config['model_only_spec_sha256'] == plan['input_sha256'][plan['model_only_source']]
            and source['mode'] == 'model_only' and source['source_completed_updates'] == baseline['completed_updates'] == (28318 if combo else 50000)
            and source['path'] == baseline['checkpoint_path'] and source['sha256'] == baseline['checkpoint_sha256']
            and source['source_training_data'] == baseline['training_data']
            and source['source_git_commit'] == baseline['training_git_commit']
            and source['source_config_sha256'] == baseline['training_config_sha256'], 'Model-only source differs')
    if combo:
        require(baseline['training_data'] == bundle['training_data']
                and baseline['evaluation_data'] == refs['dev_report']['evaluation_data']
                and baseline['protocol'] == refs['dev_report']['protocol']
                and config['mixed_image_sizes'] == [768, 896] and config['point_stats_interval'] == 250
                and config['lr_schedule'] == 'cosine' and config['learning_rate'] == 1e-5
                and config['warmup_updates'] == 300 and config['seed'] == 43
                and {k: v['targeted_draws'] for k, v in receipt['pools'].items()}
                == {'shoe': 7080, 'sock': 3540, 'belt': 3540}, 'Combined recipe or baseline differs')
    else:
        require(refs['dev_report'] == baseline, 'Historical baseline changed')
    require(baseline['evaluation_config_sha256'] == config_sha256(evaluation)
            and baseline['evaluated_images'] == bundle['dev_images'] == 250 and bundle['dev_instances'] == 1008
            and all(config[k] == v for k, v in bundle['training_data'].items()),
            'Baseline dev or training data differs')
    require(updates == config['checkpoint_updates'] == [14159, 28318]
            and config['max_updates'] == updates[-1] and config['effective_batch_images'] == 2
            and config['training_images'] == receipt['uniform_draws'] == receipt['distinct_images'] == bundle['train_images'] == 42476
            and spec['target_draws'] == receipt['targeted_draws'] == 14160
            and receipt['total_draws'] == config['max_updates'] * 2
            and spec['protection_sha256'] == receipt['protection_sha256'] == bundle['references']['protection']['sha256']
            and spec['order_sha256'] == receipt['order_sha256'], 'Sampling contract differs')
    require(plan['total_hard_seconds'] == config['budget_seconds'] == (28800 if combo else 18000) and
            plan['automatic_extension'] is False and plan['automatic_restart'] is False, 'Full wall budget differs')
    output = Path(plan['future_output'])
    runtime_config = dict(config, training_authorized=True)
    stages, previous = [], None
    for update in updates:
        name = f'update_{update:06d}'
        train, dev, previous = stage_commands(plan, output/name, update, previous, output/'launch_config.json')
        stages.append(dict(stage_name=name, new_run_update=update, train=train, full_dev=dev))
    context = dict(training_config_sha256=config_sha256(runtime_config), evaluation_config_sha256=config_sha256(evaluation),
                   training_data=bundle['training_data'], evaluation_data=bundle['evaluation_data'], model_only_source=source,
                   train_images=bundle['train_images'], train_instances=bundle['train_instances'], launch_config=runtime_config)
    prepared = dict(profile=plan['profile'], status='prepared_not_started', execution_enabled=authorized,
                    stages=stages, total_hard_seconds=plan['total_hard_seconds'],
                    per_segment_hard_seconds=plan['total_hard_seconds'],
                    baseline_AP=baseline['metrics']['AP'])
    return prepared, baseline, context


def select_full(baseline, reports, planned_updates, profile='full_single_line'):
    require([r['completed_updates'] for r in reports] == planned_updates[:len(reports)], 'Unexpected checkpoints')
    best = choose_checkpoint(reports)
    incumbent = choose_checkpoint([baseline])
    common = ('protocol', 'metric_units', 'evaluation_config_sha256', 'evaluation_data', 'evaluated_images')
    require(all(r[k] == baseline[k] for r in reports for k in common), 'Full dev protocol changed')
    improved = best['dev_mask_AP'] > incumbent['dev_mask_AP']
    winner = next(r for r in reports if r['checkpoint_sha256'] == best['checkpoint_sha256']) if improved else baseline
    cached_origin = 'cached_28318' if profile == 'full_combo' else 'cached_50k'
    return dict(selected_origin=profile if improved else cached_origin,
                selected_checkpoint=best if improved else incumbent, new_run_best=best, cached_baseline=incumbent,
                stop_reason='max_updates' if len(reports) == len(planned_updates) else 'continue',
                selected_minus_baseline={k: winner['metrics'][k] - baseline['metrics'][k] for k in ('AP', 'AP75', 'AP_small')})
