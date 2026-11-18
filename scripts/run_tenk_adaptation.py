"""Prepare by default. Explicit launch uses the profile's single shared GPU deadline."""
import time
STARTED = time.monotonic()

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from prepare_tenk_adaptation import prepare, ROOT, PLAN
from run_expansion_control import arm_deadline
from run_min_scale_trial import run_bounded

COORDINATION = Path('/root/autodl-tmp/fashion-grounding/coordination')
TOTAL_SECONDS, SEGMENT_SECONDS, CLEANUP_SECONDS = 28800, 7200, 15
EVALUATION_FILES = ['src/fashion_grounding/segmentation/' + name + '.py'
                    for name in ('evaluate', 'evaluation', 'model')]


def validate(plan, root=ROOT, *, authorized=False):
    from fashion_grounding.data.schema import require
    from fashion_grounding.segmentation.tenk_execution import read
    from fashion_grounding.segmentation.training_state import read_config
    from fashion_grounding.segmentation.evaluation import choose_checkpoint
    if plan.get('profile') == 'full_single_line':
        from full_single_plan import prepare_full
        return prepare_full(plan, root, authorized)
    required_pins = {plan[k] for k in ('training_config', 'model_only_source', 'input_validation_bundle',
                                     'evaluation_config', 'visibility_decision', 'baseline_report')}
    require(required_pins <= set(plan['input_sha256']), 'Missing execution input pin')
    prepared = prepare(plan, root, authorized=authorized)
    budget, stopping = plan['budget_proposal'], plan['stopping']
    require((budget['total_hard_seconds'], budget['per_segment_hard_seconds'],
             budget['training_internal_seconds'], budget['evaluation_internal_seconds'])
            == (TOTAL_SECONDS, SEGMENT_SECONDS, 7100, 600), 'Execution budget contract changed')
    require(stopping['max_equivalent_epochs'] == 10
            and stopping['minimum_epochs_before_plateau_stop'] == 4
            and stopping['consecutive_dev_evaluations_without_new_best'] == 3
            and stopping['automatic_restart'] is False and stopping['automatic_extension'] is False,
            'Execution stopping contract changed')
    config = read_config(root/plan['training_config'])
    evaluation = read(root/plan['evaluation_config'])
    require(config['max_updates'] == 50000 and config['effective_batch_images'] == 2
            and config['training_images'] == 10000 and config['budget_seconds'] == 7100
            and not config.get('target_crop') and evaluation['budget_seconds'] == 600,
            'Execution exposure/internal budget changed')
    bundle = read(root/plan['input_validation_bundle'])
    require((bundle['dev_images'], bundle['dev_instances'], bundle['train_instances']) == (250, 1008, 39699),
            'Execution population differs')
    baseline, source = read(root/plan['baseline_report']), read(root/plan['model_only_source'])
    choose_checkpoint([baseline])
    require(baseline['evaluated_images'] == 250 and baseline['checkpoint_path'] == source['path']
            and baseline['completed_updates'] == source['source_completed_updates'] == 20000,
            'Wrong cached C5k baseline')
    context = dict(training_config_sha256=prepared['training_config_sha256'],
        evaluation_config_sha256=prepared['evaluation_config_sha256'],
        training_data=bundle['training_data'], evaluation_data=bundle['evaluation_data'],
        model_only_source=source)
    prepared.update(execution_entry='scripts/run_tenk_adaptation.py', execution_enabled=authorized,
        total_hard_seconds=TOTAL_SECONDS, per_segment_hard_seconds=SEGMENT_SECONDS,
        early_stop='3 consecutive complete dev evaluations without exceeding incumbent; earliest epoch6',
        selection='strict new-run choice, then compare cached C5k20k on the same dev; ties keep baseline')
    return prepared, baseline, context


def source_guard(root, commit, plan_path, plan_sha, pins):
    from fashion_grounding.data.schema import sha256
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=root, text=True).strip()
    if git('rev-parse', 'HEAD') != commit or git('status', '--porcelain'):
        raise ValueError('Execution source changed or is dirty')
    if sha256(plan_path) != plan_sha or any(sha256(root/p) != v for p, v in pins.items()):
        raise ValueError('Execution plan/input changed')


def execute_segments(prepared, baseline, output, total_deadline, *, run_phase, train_receipt,
                     eval_receipt, guard, arm=arm_deadline, clock=time.monotonic):
    """Single worker process group. All callbacks are covered by the same timer."""
    from fashion_grounding.segmentation.tenk_execution import write_atomic, select_progress
    output = Path(output)
    full = prepared.get('profile') == 'full_single_line'
    segment_limit = prepared.get('per_segment_hard_seconds', SEGMENT_SECONDS)
    reports, epochs, records, previous = [], [], [], None
    for stage in prepared['stages']:
        segment_start = clock()
        deadline = min(total_deadline, segment_start + segment_limit)
        if segment_start >= deadline:
            raise TimeoutError('No total budget remains')
        arm(deadline)  # Re-arms only the segment cap; the absolute total never changes.
        name = stage['stage_name'] if full else f"epoch_{stage['equivalent_epoch']:02d}"
        directory = output/name
        directory.mkdir(exist_ok=False)
        record = dict(update=stage['new_run_update'],
                      status='running', deadline_monotonic=deadline, phases=[])
        if not full:
            record['equivalent_epoch'] = stage['equivalent_epoch']
        records.append(record)

        def persist():
            write_atomic(output/'stages.json', records)

        def check():
            if clock() >= deadline:
                raise TimeoutError('Segment/total deadline expired')

        try:
            persist()
            for phase in ('train', 'full_dev', 'select'):
                check()
                guard()
                check()
                row = dict(phase=phase, status='running')
                record['phases'].append(row)
                persist()
                if phase == 'train':
                    run_phase(stage['train'], directory/'train.log')
                    checkpoint = train_receipt(directory/'train', stage, previous)
                elif phase == 'full_dev':
                    run_phase(stage['full_dev'], directory/'full_dev.log')
                    report = eval_receipt(directory/'dev', checkpoint)
                else:
                    if full:
                        from full_single_plan import select_full
                        selection = select_full(baseline, reports+[report], [s['new_run_update'] for s in prepared['stages']])
                    else:
                        selection = select_progress(baseline, reports+[report], epochs+[stage['equivalent_epoch']])
                    check()
                    write_atomic(directory/'selection.json', selection)
                    write_atomic(output/'selection.json', selection)
                check()
                row['status'] = 'completed'
                persist()
            record.update(status='completed', elapsed_seconds=clock()-segment_start)
            persist()
            reports.append(report)
            if not full:
                epochs.append(stage['equivalent_epoch'])
            previous = checkpoint
            # Cover the gap between segments/final reporting too.
            arm(total_deadline)
            if selection['stop_reason'] != 'continue':
                return selection
        except BaseException as error:
            record.update(status='failed', error=type(error).__name__+': '+str(error))
            persist()
            raise
    raise RuntimeError('Plan exhausted without a terminal selection')


def worker(args):
    # No standalone worker bypass: the launcher supplies these pins under lock.
    if (not args.execute or os.getpid() != os.getpgrp()
            or os.environ.get('TENK_WORKER_PLAN_SHA') != args.plan_sha
            or not args.plan_sha or not args.commit or not args.deadline
            or args.deadline > time.monotonic()+TOTAL_SECONDS):
        raise ValueError('Worker requires the bounded, locked launcher')
    arm_deadline(args.deadline)  # Includes validation/import/load/selection time.
    from fashion_grounding.segmentation.tenk_execution import (read, write_atomic,
        audit_training, audit_evaluation)
    plan_path = Path(args.plan).resolve()
    plan = read(plan_path)
    prepared, baseline, context = validate(plan, authorized=True)
    context['git_commit'] = args.commit
    def guard():
        if 'launch_config' in context and read(Path(plan['future_output'])/'launch_config.json') != context['launch_config']:
            raise ValueError('Resolved launch configuration changed')
        source_guard(ROOT, args.commit, plan_path, args.plan_sha, plan['input_sha256'])
    guard()
    output = Path(plan['future_output'])
    def run_phase(command, log):
        with log.open('x') as stream:
            # Deliberately inherit the isolated worker's group: timeout/parent
            # death kills both the trainer/evaluator and its descendants.
            subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    selection = execute_segments(prepared, baseline, output, args.deadline,
        run_phase=run_phase, guard=guard,
        train_receipt=lambda d, s, p: audit_training(d, s, p, context),
        eval_receipt=lambda d, p: audit_evaluation(d, p, context))
    guard()
    write_atomic(output/'worker_result.json', dict(status='completed', selection=selection))


def launch(plan_path, plan, prepared, baseline, context, started=STARTED):
    from fashion_grounding.data.schema import sha256
    from fashion_grounding.segmentation.tenk_execution import read, write_atomic
    total_limit = prepared.get('total_hard_seconds', TOTAL_SECONDS)
    deadline = started + total_limit
    if sys.platform != 'linux' or ROOT.resolve() != Path(plan['execution_checkout']).resolve():
        raise ValueError('Execute only in the dedicated Linux segmentation checkout')
    if Path(sys.executable).resolve() != Path(plan['python']).resolve():
        raise ValueError('Use the dedicated segmentation Python runtime')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    plan_sha = sha256(plan_path)
    source_guard(ROOT, commit, plan_path, plan_sha, plan['input_sha256'])
    # Same evaluator/model implementation as the cached baseline, in addition
    # to identical config, data population and AP protocol. No old inference.
    subprocess.run(['git', 'diff', '--exit-code', baseline['evaluation_git_commit'], '--', *EVALUATION_FILES],
                   cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    output = Path(plan['future_output'])
    if not output.is_absolute() or output.exists() or not output.parent.is_dir():
        raise ValueError('Use a new absolute run output under an existing parent')
    if shutil.disk_usage(output.parent).free < 20*1024**3:
        raise ValueError('Need at least 20GiB free for retained checkpoints/predictions')
    for path in (plan['training_package'], plan['evaluation_package'], plan['upstream_source'], context['model_only_source']['path']):
        if not Path(path).exists():
            raise FileNotFoundError(path)
    # Leave time for the wrapper to record failure and release the GPU lock.
    worker_deadline = deadline-CLEANUP_SECONDS
    if time.monotonic() >= worker_deadline:
        raise TimeoutError('Total budget exhausted before launch')
    output.mkdir(exist_ok=False)
    record = dict(record_schema_version=1, status='running', owner='segmentation', executed_by='lcy',
        experiment_id='S0', split='train', candidate_source='none', seed=42,
        run_id=output.name, git_commit=commit,
        plan_sha256=plan_sha, started_at=datetime.now(timezone.utc).isoformat(),
        total_hard_seconds=total_limit, per_segment_hard_seconds=prepared.get('per_segment_hard_seconds', SEGMENT_SECONDS),
        deadline_monotonic=deadline, worker_deadline_monotonic=worker_deadline,
        cleanup_reserve_seconds=CLEANUP_SECONDS, input_sha256=plan['input_sha256'],
        training_authorized=True, automatic_restart=False, automatic_extension=False,
        model_only_source=context['model_only_source'], training_config_sha256=context['training_config_sha256'],
        evaluation_config_sha256=context['evaluation_config_sha256'], training_data=context['training_data'],
        evaluation_data=context['evaluation_data'], cached_baseline_report=plan['baseline_report'],
        evaluator_code_unchanged_from=baseline['evaluation_git_commit'])
    write_atomic(output/'run_metadata.json', record)
    write_atomic(output/'prepared_plan.json', prepared)
    if 'launch_config' in context:
        write_atomic(output/'launch_config.json', context['launch_config'])
    env = dict(os.environ, PYTHONPATH=str(ROOT/'src'), PYTHONUNBUFFERED='1', TENK_WORKER_PLAN_SHA=plan_sha)
    command = [plan['python'], str(ROOT/'scripts/with_gpu_lock.py'), '--coordination', str(COORDINATION),
        '--run-id', output.name, '--output', str(output), '--seconds', str(max(1, int(worker_deadline-time.monotonic()))),
        '--', plan['python'], str(Path(__file__).resolve()), '--plan', str(plan_path), '--execute', '--worker',
        '--deadline', str(worker_deadline), '--commit', commit, '--plan-sha', plan_sha]
    try:
        with (output/'supervisor.log').open('x') as stream:
            code = run_bounded(command, deadline, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        if code != 0:
            raise RuntimeError('Locked worker failed, exit='+str(code))
        result = read(output/'worker_result.json')
        if result['status'] != 'completed' or result['selection']['stop_reason'] not in ('plateau', 'max_equivalent_epochs', 'max_updates'):
            raise ValueError('Missing terminal selection')
        record.update(status='completed', stop_reason=result['selection']['stop_reason'])
    except BaseException as error:
        record.update(status='failed', error=type(error).__name__+': '+str(error))
        raise
    finally:
        record.update(elapsed_seconds=time.monotonic()-started, finished_at=datetime.now(timezone.utc).isoformat())
        write_atomic(output/'run_metadata.json', record)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', default=str(PLAN))
    p.add_argument('--execute', action='store_true')
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--deadline', type=float, help=argparse.SUPPRESS)
    p.add_argument('--commit', help=argparse.SUPPRESS)
    p.add_argument('--plan-sha', help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.worker:
        worker(args)
        return
    plan_path = Path(args.plan).resolve()
    plan = json.loads(plan_path.read_text())
    prepared, baseline, context = validate(plan, authorized=args.execute)
    if not args.execute:
        print(json.dumps(prepared, indent=2))
        return
    launch(plan_path, plan, prepared, baseline, context)


if __name__ == '__main__':
    main()
