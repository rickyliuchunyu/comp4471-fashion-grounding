"""Prepare by default. --execute runs only B under one 45-minute deadline."""
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


ROOT = Path(__file__).resolve().parents[1]


def build_commands(plan, out, python=sys.executable):
    out = Path(out).resolve()
    train = [python, '-m', 'fashion_grounding.segmentation.train',
        '--package', plan['training_package'], '--source', plan['upstream_source'],
        '--resume', plan['parent_checkpoint_path'], '--resume-recipe', str(ROOT/plan['recipe_policy']),
        '--config', str(ROOT/plan['training_config']), '--output', str(out/'train')]
    dev = [python, '-m', 'fashion_grounding.segmentation.evaluate',
        '--package', plan['evaluation_package'], '--training-package', plan['training_package'],
        '--source', plan['upstream_source'], '--config', str(ROOT/plan['evaluation_config']),
        '--checkpoints', str(out/'train/checkpoint_020000.pth'), '--output', str(out/'dev')]
    return [('train_B', train), ('dev250_B', dev), ('compare_cached_A', [python,
        str(ROOT/'scripts/select_min_scale_trial.py'), '--plan', str(ROOT/plan['plan_path']),
        '--output', str(out)])]


def run_bounded(command, deadline, **kwargs):
    """One monotonic deadline for the whole process group; never reset per phase."""
    remaining = deadline-time.monotonic()
    if remaining <= 0:
        raise TimeoutError('Stage deadline already exhausted')
    process = subprocess.Popen(command, start_new_session=True, **kwargs)
    try:
        return process.wait(timeout=max(0, deadline-time.monotonic()))
    except BaseException:
        # Only descendants in this invocation's new session; never other GPU work.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        raise


def validate_plan(plan):
    from fashion_grounding.data.schema import read_json, sha256, require
    from fashion_grounding.segmentation.resume_recipe import audit_recipe
    from fashion_grounding.segmentation.training_state import read_config, config_sha256
    git = lambda *a: subprocess.check_output(['git', *a], cwd=ROOT, text=True).strip()
    require(not git('status', '--porcelain', '--untracked-files=no'), 'Commit tracked changes first')
    require(plan['total_hard_limit_seconds'] == 2700 and plan['automatic_restart'] is False,
            'Trial must retain the 45-minute cap without restart')
    require(plan['evaluation_updates'] == [20000] and plan['reuse_control_A'] is True,
            'Only one new B dev evaluation is permitted')
    for path, digest in plan['input_sha256'].items():
        require(sha256(ROOT/path) == digest, 'Prepared input changed: '+path)
    config = read_config(ROOT/plan['training_config'])
    policy = read_json(ROOT/plan['recipe_policy'])
    require(plan['parent_checkpoint_sha256'] == policy['parent_checkpoint_sha256'], 'Parent hash differs')
    for key in ('control_report', 'prior_best_report'):
        report = read_json(ROOT/plan[key])
        require(report['evaluation_config_sha256'] == config_sha256(read_json(ROOT/plan['evaluation_config'])),
                'Keep A evaluation settings unchanged')
    a = read_json(ROOT/plan['control_report'])
    require(a['training_git_commit'] == policy['control_git_commit']
            and a['training_config_sha256'] == policy['control_config_sha256'], 'Wrong control A')
    a_train, a_resume = (read_json(ROOT/plan[k]) for k in ('control_train_report', 'control_resume_check'))
    require(a_train['status'] == 'completed' and a_train['start_update'] == 18000
            and a_train['completed_updates'] == 20000 and a_train['updates_this_process'] == 2000
            and a_train['git_commit'] == a['training_git_commit']
            and a_train['config_sha256'] == a['training_config_sha256']
            and a_train['checkpoints'] == [dict(update=20000, path='checkpoint_020000.pth',
                                               sha256=a['checkpoint_sha256'])], 'Control A trajectory differs')
    require(a_resume['status'] == 'passed' and a_resume['resumed_from_update'] == 18000
            and a_resume['checkpoint_sha256'] == plan['parent_checkpoint_sha256']
            and a_resume['expected_digests'] == a_resume['restored_digests']
            and a_resume['exact_components'] == ['model', 'optimizer', 'rng', 'sampler', 'scaler', 'scheduler'],
            'Control A did not restore the common parent')
    best = read_json(ROOT/plan['prior_best_report'])
    require(best['checkpoint_path'] == plan['parent_checkpoint_path']
            and best['checkpoint_sha256'] == plan['parent_checkpoint_sha256'], 'Wrong common parent')
    record = audit_recipe(ROOT, git('rev-parse', 'HEAD'), config, ROOT/plan['recipe_policy'])
    return record


def worker(plan, out):
    if os.environ.get('SEG_MIN_SCALE_WORKER') != plan['run_id']:
        raise RuntimeError('Use --execute to retain the shared lock and outer deadline')
    arm_worker_deadline(float(os.environ['SEG_MIN_SCALE_DEADLINE']))
    records = []
    for phase, command in build_commands(plan, out):
        record = dict(phase=phase, command=command, status='running',
                      started_at=datetime.now(timezone.utc).isoformat())
        records.append(record)
        (out/'phases.json').write_text(json.dumps(records, indent=2)+'\n')
        print(json.dumps(record), flush=True)
        # Stay in the wrapper's process group. The one outer deadline covers all.
        code = subprocess.call(command, cwd=ROOT)
        record.update(status='completed' if code == 0 else 'failed', exit_code=code)
        (out/'phases.json').write_text(json.dumps(records, indent=2)+'\n')
        if code:
            return code
    return 0


def arm_worker_deadline(deadline):
    # with_gpu_lock starts this worker as its own process-group leader. A timer
    # inside that group also kills training if the supervising wrapper disappears.
    if os.getpgrp() != os.getpid():
        raise RuntimeError('Worker must own its isolated process group')
    remaining = deadline-time.monotonic()
    if remaining <= 0:
        raise TimeoutError('Worker deadline exhausted')
    def expire(signum, frame):
        os.killpg(os.getpgrp(), signal.SIGKILL)
    signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, remaining)


def main():
    started = time.monotonic()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', default='configs/runs/segmentation/min_scale_05_stage_plan.json')
    p.add_argument('--output', required=True)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = p.parse_args()
    plan_path, out = Path(args.plan).resolve(), Path(args.output).resolve()
    plan = json.loads(plan_path.read_text())
    if plan_path != (ROOT/plan['plan_path']).resolve():
        raise ValueError('Use the pinned stage plan')
    if args.worker:
        raise SystemExit(worker(plan, out))
    record = validate_plan(plan)
    if not args.execute:
        print(json.dumps(dict(status='prepared_not_started', metadata_audit=record,
            hard_limit_seconds=2700, commands=build_commands(plan, out),
            model_loads=0, optimizer_updates=0, live_restore_verified=False), indent=2))
        return
    deadline = started+2700
    if sys.platform != 'linux' or not out.parent.is_dir():
        raise RuntimeError('Execute only on the prepared Linux server, with existing output parent')
    if shutil.disk_usage(out.parent).free < 10*1024**3:
        raise OSError('Need 10 GiB free for the new checkpoint, dev predictions and logs')
    out.mkdir(exist_ok=False)
    env = dict(os.environ, PYTHONPATH=str(ROOT/'src'), SEG_MIN_SCALE_WORKER=plan['run_id'],
               SEG_MIN_SCALE_DEADLINE=str(deadline-10))
    seconds = int(deadline-time.monotonic())-10  # leave the existing wrapper five seconds for cleanup
    if seconds <= 0:
        raise TimeoutError('Preparation exhausted the stage budget')
    command = [sys.executable, str(ROOT/'scripts/with_gpu_lock.py'),
        '--coordination', plan['coordination'], '--run-id', plan['run_id'],
        '--output', str(out), '--seconds', str(seconds), '--', sys.executable,
        str(Path(__file__).resolve()), '--plan', str(plan_path), '--output', str(out), '--worker']
    status = dict(status='running', started_at=datetime.now(timezone.utc).isoformat(),
                  hard_limit_seconds=2700, command=command, metadata_audit=record)
    target = out/'stage_status.json'
    target.write_text(json.dumps(status, indent=2)+'\n')
    try:
        with (out/'stage.log').open('x') as log:
            code = run_bounded(command, deadline, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        status.update(status='completed' if code == 0 else 'failed', exit_code=code)
    except BaseException as error:
        status.update(status='failed', error=type(error).__name__)
        raise
    finally:
        status['elapsed_seconds'] = time.monotonic()-started
        status['finished_at'] = datetime.now(timezone.utc).isoformat()
        target.write_text(json.dumps(status, indent=2)+'\n')
    raise SystemExit(code)


if __name__ == '__main__':
    main()
