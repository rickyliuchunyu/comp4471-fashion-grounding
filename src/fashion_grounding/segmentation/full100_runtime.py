"""Sequential train/dev orchestration with explicit resume and a shared GPU lock."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

from .full100_contract import (choose, digest_json, is_sha, read_json, require,
                               sha256, validate_report)


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + f'.{os.getpid()}.partial')
    with tmp.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, path)


def checkpoints(root, contract):
    root = Path(root).resolve()
    found = {}
    for path in sorted(root.glob('attempts/*/train/checkpoint_*.receipt.json')):
        row = read_json(path)
        target = Path(row['path']).resolve()
        require(target.is_relative_to(root) and target.parent == path.parent.resolve(), 'Checkpoint path escaped run')
        require(type(row['update']) is int and row['update'] in contract['schedule']['checkpoint_updates'],
                'Unknown checkpoint node')
        require(row['training_config_sha256'] == contract['train_config_sha256']
                and row['source_commit'] == contract['source_commit']
                and row['binding_sha256'] == contract['inputs']['binding_sha256'], 'Checkpoint receipt lineage differs')
        require(is_sha(row['sha256']) and target.is_file()
                and target.with_suffix('.sha256').read_text().strip() == row['sha256'], 'Incomplete checkpoint artifact')
        stat = target.stat()
        require(stat.st_size == row['bytes'] and stat.st_mtime_ns == row['mtime_ns'], 'Checkpoint file changed')
        require(row['update'] not in found or found[row['update']]['sha256'] == row['sha256'],
                'Conflicting checkpoints at one update')
        found[row['update']] = row
    return found


def reports(root, contract, saved):
    found = {}
    targets = {node['update'] for node in contract['schedule']['evaluation_nodes']}
    for path in sorted(Path(root).glob('attempts/*/evaluate/evaluation_report.json')):
        # A crash before publishing the report sidecar leaves an incomplete attempt.
        if not path.with_suffix('.sha256').exists():
            continue
        require(sha256(path) == path.with_suffix('.sha256').read_text().strip(), 'Evaluation report changed')
        report = read_json(path)
        update = report['completed_updates']
        require(update in targets and update in saved, 'Evaluation has no scheduled checkpoint')
        validate_report(report, contract, saved[update])
        predictions = path.parent/'predictions.json'
        require(predictions.is_file() and sha256(predictions) == report['predictions_sha256'],
                'Cached predictions changed')
        require(update not in found, 'Multiple completed evaluations for one checkpoint')
        found[update] = report
    return found


def run_engine(contract, output, execute_phase, *, resume=False):
    """execute_phase(mode, output, stop, resume_checkpoint, eval_checkpoint).

    Tests supply CPU artifact fixtures. Production exclusively invokes GPU workers.
    A failed phase stops immediately; only a later explicit --resume continues it.
    """
    require(contract['ready_for_training'] is True, 'Full-data evidence is not ready')
    output = Path(output).resolve()
    if resume:
        require((output/'contract.json').is_file() and read_json(output/'contract.json') == contract,
                'Cannot resume a different or missing run contract')
    else:
        output.mkdir(parents=True, exist_ok=False)
        atomic_json(output/'contract.json', contract)
        (output/'attempts').mkdir()
    state = dict(status='running', source_commit=contract['source_commit'],
                 contract_sha256=digest_json(contract), resumed=resume, updated_at=now())
    atomic_json(output/'stage_status.json', state)
    try:
        saved = checkpoints(output, contract)
        complete = reports(output, contract, saved)
        attempt = max([int(p.name) for p in (output/'attempts').iterdir() if p.is_dir()] or [0])

        def phase(mode, stop, resume_checkpoint=None, eval_checkpoint=None):
            nonlocal attempt
            attempt += 1
            directory = output/'attempts'/f'{attempt:06d}'
            directory.mkdir(exist_ok=False)
            record = dict(mode=mode, target_update=stop, status='running', started_at=now(),
                          resume_checkpoint=resume_checkpoint, eval_checkpoint=eval_checkpoint)
            atomic_json(directory/'phase.json', record)
            try:
                execute_phase(mode, directory/mode, stop, resume_checkpoint, eval_checkpoint)
                record.update(status='completed', finished_at=now())
            except BaseException as error:
                record.update(status='failed', finished_at=now(), error=type(error).__name__+': '+str(error))
                raise
            finally:
                atomic_json(directory/'phase.json', record)

        for node in contract['schedule']['evaluation_nodes']:
            update = node['update']
            if update not in saved:
                require(not saved or max(saved) < update, 'Missing earlier checkpoint; do not rewind training silently')
                previous = saved[max(saved)]['path'] if saved else None
                phase('train', update, resume_checkpoint=previous)
                saved = checkpoints(output, contract)
                require(update in saved, 'Training phase did not publish the requested checkpoint')
            if update not in complete:
                phase('evaluate', update, eval_checkpoint=saved[update]['path'])
                complete = reports(output, contract, saved)
                require(update in complete, 'Evaluation phase did not publish a complete dev report')
            selection = choose([complete[n['update']] for n in contract['schedule']['evaluation_nodes']
                                if n['update'] <= update])
            selection.update(trajectory_complete=update == contract['schedule']['max_updates'])
            atomic_json(output/'selection.json', selection)
            state.update(last_evaluated_epoch=node['epoch'], last_evaluated_update=update,
                         latest_recovery_update=max(saved), completed_evaluations=len(complete), updated_at=now())
            atomic_json(output/'stage_status.json', state)
        winner = read_json(output/'selection.json')
        require(sha256(winner['checkpoint_path']) == winner['checkpoint_sha256'], 'Selected checkpoint integrity failure')
        require(len(complete) == len(contract['schedule']['evaluation_nodes']), 'Incomplete evaluation trajectory')
        state.update(status='completed', finished_at=now())
    except BaseException as error:
        state.update(status='interrupted' if isinstance(error, (KeyboardInterrupt, SystemExit)) else 'failed',
                     error=type(error).__name__+': '+str(error), finished_at=now())
        raise
    finally:
        atomic_json(output/'stage_status.json', state)
    return state


def _parent_death_guard(parent):
    import ctypes
    if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        raise OSError('Cannot install Linux parent-death protection')
    if os.getppid() != parent:
        os.kill(os.getpid(), signal.SIGKILL)


def run_process(command, log_path, cwd, environment):
    """No total time cap; terminate only our child process group on interruption."""
    parent = os.getpid()
    with Path(log_path).open('x') as log:
        process = subprocess.Popen(command, cwd=cwd, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True,
                                   preexec_fn=(lambda: _parent_death_guard(parent)) if sys.platform == 'linux' else None)
        try:
            code = process.wait()
            if code:
                raise RuntimeError(f'Worker exited {code}; see {log_path}')
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


def worker_command(contract, root, mode, output, stop, resume_checkpoint, eval_checkpoint):
    command = [contract['runtime']['python'], '-m', 'fashion_grounding.segmentation.full100_worker',
               '--contract', str(Path(root)/'contract.json'), '--mode', mode, '--output', str(output)]
    if mode == 'train':
        command += ['--stop-update', str(stop)]
        if resume_checkpoint:
            command += ['--resume', resume_checkpoint]
    else:
        command += ['--checkpoint', eval_checkpoint]
    return command


def pid_exists(pid):
    if type(pid) is not int or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


@contextmanager
def gpu_slot(contract, output, resume=False):
    import fcntl
    require(sys.platform == 'linux', 'GPU execution is supported only on the Linux training server')
    output = Path(output).resolve()
    policy = contract['runtime']
    allowed = Path(policy['runs_root']).resolve()
    require(output.parent == allowed and output.name.startswith('segmentation-full100-'), 'Use a new full100 run directory')
    require((output.exists() and resume) or (not output.exists() and not resume), 'Run directory/resume mode mismatch')
    coordination = Path(policy['coordination'])
    coordination.mkdir(parents=True, exist_ok=True)
    with (coordination/'gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
        memory = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
        require(not processes and all(int(v) <= 256 for v in memory.splitlines()), 'GPU occupied; leave current jobs untouched')
        for path in coordination.glob('*.json'):
            record = read_json(path)
            if record.get('status') != 'running':
                continue
            same = (resume and record.get('entry') == 'full100' and record.get('output') == str(output)
                    and record.get('contract_sha256') == digest_json(contract))
            require(same and type(record.get('wrapper_pid')) is int and record['wrapper_pid'] > 0
                    and not pid_exists(record['wrapper_pid']), 'Unresolved running coordination record: '+path.name)
            record.update(status='interrupted', reconciled_at=now(), reconciliation='explicit own-run resume; owner dead and GPU empty')
            atomic_json(path, record)
        existing = len(checkpoints(output, contract)) if resume else 0
        required = policy['disk_reserve_bytes'] + (len(contract['schedule']['checkpoint_updates'])-existing)*policy['estimated_checkpoint_bytes']
        require(shutil.disk_usage(allowed).free >= required, 'Insufficient planned checkpoint/metric disk space')
        run_id = output.name+'-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
        path = coordination/(run_id+'.json')
        record = dict(entry='full100', role='segmentation', executed_by='lcy', run_id=run_id,
                      output=str(output), status='running', wrapper_pid=os.getpid(), started_at=now(),
                      contract_sha256=digest_json(contract), source_commit=contract['source_commit'],
                      total_wall_seconds=None, automatic_restart=False, explicit_resume=resume)
        atomic_json(path, record)
        try:
            yield path
            record.update(status='completed', finished_at=now())
        except BaseException as error:
            record.update(status='failed', error=type(error).__name__+': '+str(error), finished_at=now())
            raise
        finally:
            atomic_json(path, record)


def execute(contract, output, repo_root, resume=False):
    output = Path(output).resolve()
    previous = signal.getsignal(signal.SIGTERM)
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Termination requested')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        with gpu_slot(contract, output, resume=resume) as coordination_record:
            environment = dict(os.environ, PYTHONPATH=str(Path(repo_root)/'src'),
                               FULL100_COORDINATION_RECORD=str(coordination_record))
            def phase(mode, directory, stop, resume_checkpoint, eval_checkpoint):
                command = worker_command(contract, output, mode, directory, stop, resume_checkpoint, eval_checkpoint)
                run_process(command, directory.parent/(mode+'.log'), repo_root, environment)
            return run_engine(contract, output, phase, resume=resume)
    finally:
        signal.signal(signal.SIGTERM, previous)
