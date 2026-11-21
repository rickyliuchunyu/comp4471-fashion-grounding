"""One dev250 candidate pass, then CPU exports. Outer timeout: 1800 seconds.

No automatic restart. GPU wrapper has an independent 900-second bound; it
releases the shared lock before the CPU-only consumer phase. The outer timeout
allows five seconds of termination cleanup and never extends the work budget.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    run, repo = map(Path, sys.argv[1:])
    root = Path('/root/autodl-tmp/fashion-grounding')
    started = time.monotonic()
    env = dict(os.environ, PYTHONPATH=str(repo/'src'), OMP_NUM_THREADS='2',
               MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    config = str(repo/'configs/runs/segmentation/candidate_u18000_export.json')
    gpu = [sys.executable, str(repo/'scripts/with_gpu_lock.py'),
           '--coordination', str(root/'coordination'), '--run-id', run.name,
           '--output', str(run/'cache'), '--seconds', '900', '--',
           sys.executable, '-m', 'fashion_grounding.segmentation.export_candidates',
           'cache', '--config', config, '--split', 'dev',
           '--package', str(root/'shared/processed/fp_dev250_v1'),
           '--checkpoint', str(root/'runs/segmentation/segmentation-s0-20000-20261115-r01/train/checkpoint_018000.pth'),
           '--source', str(root/'vendor/segmentation/Mask2Former'),
           '--run-id', run.name, '--output', str(run/'cache'), '--budget-seconds', '850']
    cpu = ['timeout', '--signal=TERM', '--kill-after=5s', '900s',
           sys.executable, str(run/'export_and_accept.py'), str(run), str(repo)]
    record = dict(run_id=run.name, status='running',
                  started_at=datetime.now(timezone.utc).isoformat(),
                  hard_limit_seconds=1800, cleanup_grace_seconds=5,
                  git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
                  phases=[], automatic_retries=0, optimizer_updates=0)
    path = run/'stage_receipt.json'
    with path.open('x') as f:
        json.dump(record, f, indent=2)
    try:
        for name, command in [('cache', gpu), ('offline', cpu)]:
            phase = dict(name=name, status='running', command=command,
                         started_at=datetime.now(timezone.utc).isoformat())
            record['phases'].append(phase)
            path.write_text(json.dumps(record, indent=2)+'\n')
            print(json.dumps(phase), flush=True)
            phase_start = time.monotonic()
            with (run/(name+'.log')).open('x') as log:
                code = subprocess.call(command, cwd=repo,
                    env=dict(env, CUDA_VISIBLE_DEVICES='' if name == 'offline' else '0'),
                    stdout=log, stderr=subprocess.STDOUT)
            phase.update(exit_code=code, status='completed' if code == 0 else 'failed',
                         elapsed_seconds=time.monotonic()-phase_start,
                         finished_at=datetime.now(timezone.utc).isoformat())
            path.write_text(json.dumps(record, indent=2)+'\n')
            if code:
                raise RuntimeError('Phase failed: '+name)
        record['status'] = 'completed'
    except BaseException as error:
        record.update(status='failed', error_type=type(error).__name__)
        raise
    finally:
        record.update(elapsed_seconds=time.monotonic()-started,
                      finished_at=datetime.now(timezone.utc).isoformat())
        path.write_text(json.dumps(record, indent=2)+'\n')


if __name__ == '__main__':
    main()
