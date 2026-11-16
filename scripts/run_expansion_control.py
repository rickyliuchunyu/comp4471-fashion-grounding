"""Prepare by default; --execute runs C5k then D10k with separate 3h hard caps.

Saved 10k/20k checkpoints are evaluated together after each arm's 20k training.
The shared lock, worker timer and outer supervisor cover all phases, including
receipt checks and selection. No retry, deadline reset, or automatic continuation.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from run_min_scale_trial import run_bounded, arm_worker_deadline

ROOT = Path(__file__).resolve().parents[1]
PLAN = 'configs/runs/segmentation/expansion_control_plan.json'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def validate_plan(plan):
    from fashion_grounding.data.schema import require, sha256
    from fashion_grounding.segmentation.training_state import read_config
    from fashion_grounding.segmentation.evaluation import read_eval_config
    from fashion_grounding.segmentation.input_receipts import read_evidence_bundle, validate_chain
    require(plan['plan_path'] == PLAN and plan['same_successful_updates'] == 20000
            and plan['effective_batch_images'] == 2, 'Keep the paired 20k budget')
    budget = plan['budget']
    require(budget['per_arm_hard_seconds'] == 10800 and budget['total_pair_hard_seconds'] == 21600
            and budget['automatic_extension'] is False and budget['automatic_restart'] is False,
            'Keep 3h per arm, no extension/restart')
    require(plan['evaluation']['updates'] == [10000, 20000]
            and plan['evaluation']['primary_endpoint'] == 20000, 'Keep the two evaluation points')
    require([a['name'] for a in plan['arms']] == ['C5k', 'D10k_targeted'], 'Wrong pair/order')
    require(len({a['run_id'] for a in plan['arms']}) == 2
            and all(Path(a['run_id']).name == a['run_id'] for a in plan['arms']), 'Unsafe or duplicate run IDs')
    for path, expected in plan['input_sha256'].items():
        require(sha256(ROOT/path) == expected, 'Pinned input changed: '+path)
    configs, receipts = [], {}
    for arm in plan['arms']:
        cfg = read_config(ROOT/arm['training_config'])
        bundle, records, _ = read_evidence_bundle(ROOT/arm['input_validation_bundle'])
        validate_chain(bundle, records)
        require(bundle['training_data'] == {k: cfg[k] for k in bundle['training_data']}, 'Arm receipt/config mismatch')
        require(cfg['max_updates'] == 20000 and cfg['effective_batch_images'] == 2
                and cfg['seed'] == 42 and cfg['min_scale'] == .5
                and cfg['budget_seconds'] == budget['training_internal_seconds']
                and set(plan['evaluation']['updates']) <= set(cfg['checkpoint_updates']), 'Wrong arm recipe/budget')
        configs.append(cfg)
        receipts[arm['name']] = {'mode': bundle['mode'], 'sha256': sha256(ROOT/arm['input_validation_bundle'])}
    differences = {k for k in configs[0].keys() | configs[1].keys() if configs[0].get(k) != configs[1].get(k)}
    require(differences == {'image_manifest_sha256', 'annotation_sha256'}, 'Non-data factor changed')
    evaluation = read_eval_config(ROOT/plan['evaluation']['config'])
    require((evaluation['short_edge'], evaluation['max_size']) == (800, 1333)
            and evaluation['budget_seconds'] == budget['evaluation_internal_seconds'], 'Changed dev scale/budget')
    for arm in plan['arms']:
        bundle = read(ROOT/arm['input_validation_bundle'])
        require(bundle['evaluation_data'] == {k: evaluation[k] for k in bundle['evaluation_data']}
                and bundle['dev_images'] == 250 and bundle['dev_instances'] == 1008, 'Different full dev inputs')
    provenance = read(ROOT/'envs/segmentation/upstream.json')
    require(plan['initialization']['mode'] == 'fresh_official_COCO_both_arms'
            and plan['initialization']['sha256'] == provenance['checkpoint_sha256'], 'Wrong common initialization')
    return {'different_config_keys': sorted(differences), 'receipts': receipts}


def check_live_inputs(plan, arms):
    from fashion_grounding.segmentation.input_receipts import reuse_input_receipts
    cfg = read(ROOT/plan['evaluation']['config'])
    return {arm['name']: reuse_input_receipts(plan['evaluation']['package'], cfg, arm['package'],
                                            ROOT/arm['input_validation_bundle'])[2] for arm in arms}


def commands(plan, arm, output):
    python, directory = plan['python'], Path(output)/arm['name']
    train = [python, '-m', 'fashion_grounding.segmentation.train', '--package', arm['package'],
             '--source', plan['upstream_source'], '--checkpoint', plan['initialization']['path'],
             '--config', str(ROOT/arm['training_config']), '--output', str(directory/'train')]
    dev = [python, '-m', 'fashion_grounding.segmentation.evaluate', '--package', plan['evaluation']['package'],
           '--training-package', arm['package'], '--source', plan['upstream_source'],
           '--config', str(ROOT/plan['evaluation']['config']), '--input-validation-bundle',
           str(ROOT/arm['input_validation_bundle']), '--checkpoints',
           *[str(directory/f'train/checkpoint_{u:06d}.pth') for u in plan['evaluation']['updates']],
           '--output', str(directory/'dev')]
    select = [python, str(Path(__file__).resolve()), '--plan', str(ROOT/PLAN), '--output', str(output),
              '--select-arm', arm['name']]
    return [('train_20000', train), ('full_dev_10000_20000', dev), ('select_and_compare', select)]


def arm_deadline(deadline):
    arm_worker_deadline(deadline)
    # The wrapper owns this isolated worker session. If the wrapper is killed,
    # terminate its worker AND descendants immediately, without waiting for 3h.
    def terminate_group(signum, frame):
        os.killpg(os.getpgrp(), signal.SIGKILL)
    signal.signal(signal.SIGTERM, terminate_group)
    if sys.platform == 'linux':
        import ctypes
        parent = os.getppid()
        if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
            raise OSError('Cannot install parent-death guard')
        if parent == 1 or os.getppid() != parent:
            terminate_group(None, None)


def execute_phases(phases, directory, deadline):
    """Worker-only, children stay in this session; one timer spans all phases."""
    directory = Path(directory)
    arm_deadline(deadline)
    records = []
    for name, command in phases:
        if time.monotonic() >= deadline:
            raise TimeoutError('No arm budget remains')
        row = dict(phase=name, command=command, status='running', started_at=datetime.now(timezone.utc).isoformat())
        records.append(row)
        write(directory/'phases.json', records)
        start = time.monotonic()
        with (directory/(name+'.log')).open('x') as log:
            code = subprocess.call(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        row.update(status='completed' if code == 0 else 'failed', exit_code=code,
                   elapsed_seconds=time.monotonic()-start)
        write(directory/'phases.json', records)
        if code:
            return code
    return 0


def select_arm(plan, arm, output):
    from fashion_grounding.data.schema import require, sha256
    from fashion_grounding.segmentation.training_state import read_config, config_sha256
    from fashion_grounding.segmentation.evaluation import choose_checkpoint
    directory = Path(output)/arm['name']
    train, cfg = read(directory/'train/train_report.json'), read_config(ROOT/arm['training_config'])
    head = git('rev-parse', 'HEAD')
    require(train['status'] == 'completed' and train['completed_updates'] == 20000
            and train['start_update'] == 0 and train['updates_this_process'] == 20000 and not train['resumed']
            and train['git_commit'] == head and train['config_sha256'] == config_sha256(cfg)
            and train['all_losses_and_gradients_finite'] and train['skipped_optimizer_updates'] == 0
            and train['sample_count'] == 40000,
            'Incomplete or different arm training')
    require(read(directory/'dev/run_report.json')['status'] == 'completed', 'Incomplete dev run')
    checkpoints = {r['update']: r for r in train['checkpoints']}
    reports, evidence = [], []
    for update in plan['evaluation']['updates']:
        path = directory/f'dev/update_{update:06d}/evaluation_report.json'
        digest = sha256(path)
        require(digest == path.with_suffix('.sha256').read_text().strip(), 'Report hash changed')
        r = read(path)
        require(r['completed_updates'] == update and r['evaluated_images'] == 250
                and r['evaluation_git_commit'] == r['training_git_commit'] == head
                and r['training_config_sha256'] == config_sha256(cfg)
                and r['evaluation_config_sha256'] == config_sha256(read(ROOT/plan['evaluation']['config']))
                and r['training_data'] == {k: cfg[k] for k in ('image_manifest_sha256', 'annotation_sha256')}
                and r['evaluation_data'] == read(ROOT/arm['input_validation_bundle'])['evaluation_data']
                and r['optimizer_updates'] == 0 and r['checkpoint_model_unchanged'] and not r['queries_read']
                and r['checkpoint_path'] == str(directory/f'train/checkpoint_{update:06d}.pth')
                and r['checkpoint_sha256'] == checkpoints[update]['sha256'], 'Wrong evaluation identity')
        reports.append(r)
        evidence.append(dict(path=str(path), sha256=digest))
    result = dict(status='completed', arm=arm['name'], selected=choose_checkpoint(reports), reports=evidence,
                  evaluation_updates=plan['evaluation']['updates'], source_commit=head)
    write(directory/'selection.json', result)
    return result


def compare_pair(plan, output):
    from fashion_grounding.data.schema import sha256, require
    from fashion_grounding.segmentation.evaluation import choose_checkpoint
    output = Path(output)
    controls, trials = [], []
    for arm, target in zip(plan['arms'], (controls, trials)):
        selection = read(output/arm['name']/'selection.json')
        require(selection['status'] == 'completed', 'Incomplete arm selection')
        for item in selection['reports']:
            require(sha256(item['path']) == item['sha256'], 'Selected report changed')
            target.append(read(item['path']))
        require([r['completed_updates'] for r in target] == plan['evaluation']['updates'], 'Missing paired endpoint')
        require(choose_checkpoint(target) == selection['selected'], 'Selection changed')
    a, b = controls[-1], trials[-1]
    for key in ('evaluation_data', 'evaluation_config_sha256', 'protocol', 'metric_units', 'evaluation_git_commit', 'training_git_commit'):
        require(a[key] == b[key], 'Incomparable pair: '+key)
    initial = [read(output/arm['name']/'train/train_report.json')['initial_parameter_digests'] for arm in plan['arms']]
    require(set(initial[0]) == {'backbone', 'classifier'} and initial[0] == initial[1], 'Different initial parameters')
    delta = {k: b['metrics'][k]-a['metrics'][k] for k in ('AP', 'AP_small', 'AP75')}
    weak = {str(i): b['metrics']['per_category_AP'][str(i)]-a['metrics']['per_category_AP'][str(i)] for i in (3, 4, 7, 8)}
    screen = plan['practical_screen']
    passed = (delta['AP'] >= screen['primary_delta_mask_AP_min']
              and delta['AP_small'] >= screen['delta_AP_small_min']
              and delta['AP75'] >= screen['delta_AP75_min']
              and min(weak.values()) >= -screen['max_drop_per_weak_project_category_AP'])
    historical = read(ROOT/plan['historical_report'])
    choose_checkpoint([historical])
    require(historical['evaluation_data'] == a['evaluation_data'], 'Historical dev changed')
    choices = [(historical, 'historical_B20k'), *[(r, 'C5k') for r in controls], *[(r, 'D10k_targeted') for r in trials]]
    best, group = min(choices, key=lambda pair: (-pair[0]['metrics']['AP'], choices.index(pair)))
    result = dict(status='completed', endpoint=20000, D10k_minus_C5k=delta,
                  weak_category_AP_deltas=weak, practical_screen_passed=passed,
                  best_by_dev_AP=dict(group=group, checkpoint_path=best['checkpoint_path'],
                                      checkpoint_sha256=best['checkpoint_sha256'], AP=best['metrics']['AP']),
                  historical_B20k_AP=historical['metrics']['AP'],
                  deployment_recommendation='review_D10k' if passed and group == 'D10k_targeted' else 'retain_historical_pending_review',
                  weights_or_candidates_replaced=False, statistical_significance_claim=False,
                  subgroup_boundary_diagnostics='Use saved predictions; not rerun by this launcher')
    write(output/'comparison.json', result)
    return result


def run_pair(plan, output, start, source_commit):
    from fashion_grounding.data.schema import sha256
    output = Path(output)
    if output.exists() or not output.parent.is_dir() or shutil.disk_usage(output.parent).free < 20*1024**3:
        raise RuntimeError('Need a new output under an existing parent with 20 GiB free')
    output.mkdir()
    pair_deadline = start+plan['budget']['total_pair_hard_seconds']
    state = dict(status='running', source_commit=source_commit, plan_sha256=sha256(ROOT/PLAN),
                 per_arm_hard_seconds=10800, pair_hard_seconds=21600, automatic_restart=False, arms=[])
    write(output/'stage_status.json', state)
    write(output/'resolved_plan.json', plan)
    try:
        for arm in plan['arms']:
            arm_start = time.monotonic()
            deadline = min(arm_start+10800, pair_deadline)
            if git('rev-parse', 'HEAD') != source_commit or git('status', '--porcelain'):
                raise RuntimeError('Common source changed between arms')
            directory = output/arm['name']; directory.mkdir()
            row = dict(arm=arm['name'], run_id=arm['run_id'], status='running',
                       hard_deadline_monotonic=deadline, started_at=datetime.now(timezone.utc).isoformat())
            metadata = dict(record_schema_version=1, run_id=arm['run_id'], owner='segmentation',
                            executed_by='lcy', status='running', git_commit=source_commit,
                            plan_sha256=state['plan_sha256'], training_config=arm['training_config'],
                            training_config_file_sha256=sha256(ROOT/arm['training_config']),
                            input_validation_bundle=arm['input_validation_bundle'],
                            input_validation_bundle_sha256=sha256(ROOT/arm['input_validation_bundle']),
                            training_package=arm['package'], evaluation_package=plan['evaluation']['package'],
                            seed=42, updates=20000, evaluation_updates=[10000, 20000],
                            hard_limit_seconds=10800, automatic_restart=False, automatic_extension=False,
                            started_at=row['started_at'], artifacts={})
            write(directory/'run_metadata.json', metadata)
            state['arms'].append(row); write(output/'stage_status.json', state)
            remaining = int(deadline-time.monotonic())-15
            env = dict(os.environ, PYTHONPATH=str(ROOT/'src'), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                       MKL_NUM_THREADS='1', SEG_EXPANSION_ARM=arm['run_id'], SEG_EXPANSION_DEADLINE=str(deadline-10),
                       SEG_EXPANSION_SOURCE=source_commit, SEG_EXPANSION_PLAN_SHA=state['plan_sha256'])
            lock = [plan['python'], str(ROOT/'scripts/with_gpu_lock.py'), '--coordination', plan['coordination'],
                    '--run-id', arm['run_id'], '--output', str(directory), '--seconds', str(remaining), '--',
                    plan['python'], str(Path(__file__).resolve()), '--plan', str(ROOT/PLAN), '--output', str(output),
                    '--worker-arm', arm['name']]
            try:
                if remaining <= 0:
                    raise TimeoutError('Pair deadline exhausted')
                with (directory/'supervisor.log').open('x') as log:
                    code = run_bounded(lock, deadline-5, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                row.update(exit_code=code, status='completed' if code == 0 else 'failed')
                if code:
                    raise RuntimeError('Arm failed; no next arm or automatic retry')
            except BaseException as error:
                row.update(status='failed', error=type(error).__name__)
                raise
            finally:
                row['elapsed_seconds'] = time.monotonic()-arm_start
                metadata.update(status=row['status'], elapsed_seconds=row['elapsed_seconds'],
                                exit_code=row.get('exit_code'), error=row.get('error'),
                                finished_at=datetime.now(timezone.utc).isoformat())
                metadata['artifacts'] = {name: str(directory/name) for name in
                    ('input_receipts.json', 'phases.json', 'train/train_report.json', 'dev/run_report.json', 'selection.json')
                    if (directory/name).exists()}
                write(directory/'run_metadata.json', metadata)
                write(output/'stage_status.json', state)
        state['status'] = 'completed'
    except BaseException as error:
        state.update(status='failed', error=type(error).__name__)
        raise
    finally:
        state.update(elapsed_seconds=time.monotonic()-start, finished_at=datetime.now(timezone.utc).isoformat())
        write(output/'stage_status.json', state)


def main():
    start = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', default=PLAN)
    parser.add_argument('--output', required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--worker-arm', help=argparse.SUPPRESS)
    mode.add_argument('--select-arm', help=argparse.SUPPRESS)
    parser.add_argument('--check-inputs', action='store_true', help='Metadata-only real package receipt checks; no execution')
    args = parser.parse_args()
    if Path(args.plan).resolve() != (ROOT/PLAN).resolve():
        raise ValueError('Use the pinned plan')
    plan, output = read(args.plan), Path(args.output).resolve()
    if args.worker_arm:
        from fashion_grounding.data.schema import sha256
        arm = next(a for a in plan['arms'] if a['name'] == args.worker_arm)
        if (os.environ.get('SEG_EXPANSION_ARM') != arm['run_id'] or git('rev-parse', 'HEAD') != os.environ.get('SEG_EXPANSION_SOURCE')
                or sha256(ROOT/PLAN) != os.environ.get('SEG_EXPANSION_PLAN_SHA')):
            raise RuntimeError('Worker requires the shared-lock supervisor and pinned source')
        deadline = float(os.environ['SEG_EXPANSION_DEADLINE'])
        arm_deadline(deadline)
        validate_plan(plan)
        write(output/arm['name']/'input_receipts.json', check_live_inputs(plan, [arm]))
        raise SystemExit(execute_phases(commands(plan, arm, output), output/arm['name'], deadline))
    if args.select_arm:
        from fashion_grounding.data.schema import sha256
        arm = next(a for a in plan['arms'] if a['name'] == args.select_arm)
        if (os.environ.get('SEG_EXPANSION_ARM') != arm['run_id']
                or git('rev-parse', 'HEAD') != os.environ.get('SEG_EXPANSION_SOURCE')
                or sha256(ROOT/PLAN) != os.environ.get('SEG_EXPANSION_PLAN_SHA')
                or git('status', '--porcelain')):
            raise RuntimeError('Selection belongs to the bounded worker')
        select_arm(plan, arm, output)
        if arm['name'] == 'D10k_targeted':
            compare_pair(plan, output)
        return
    audit = validate_plan(plan)
    if not args.execute:
        print(json.dumps(dict(status='prepared_not_started', audit=audit,
            live_input_receipts=check_live_inputs(plan, plan['arms']) if args.check_inputs else None,
            per_arm_hard_seconds=10800, pair_hard_seconds=21600, model_loads=0, inference_images=0,
            training_updates=0, commands={a['name']: commands(plan, a, output) for a in plan['arms']}), indent=2))
        return
    if args.check_inputs:
        parser.error('--check-inputs is a preparation-only mode')
    if sys.platform != 'linux' or Path(sys.executable).resolve() != Path(plan['python']).resolve():
        raise RuntimeError('Execution requires the pinned Linux segmentation environment')
    if git('status', '--porcelain'):
        raise RuntimeError('Commit source before execution')
    run_pair(plan, output, start, git('rev-parse', 'HEAD'))


if __name__ == '__main__':
    main()
