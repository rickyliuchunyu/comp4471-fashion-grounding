"""Linux shared-GPU entry point. Lock is advisory; all roles must adopt it."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--coordination', required=True)
    p.add_argument('--run-id', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--seconds', required=True, type=int)
    p.add_argument('command', nargs=argparse.REMAINDER)
    args = p.parse_args()
    if not args.command or args.seconds <= 0 or Path(args.run_id).name != args.run_id:
        p.error('Need safe run ID, positive timeout, and a command after --')
    command = args.command[1:] if args.command[0] == '--' else args.command
    root = Path(args.coordination)
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for path in root.glob('*.json'):
            record = json.loads(path.read_text())
            if record.get('status') == 'running':
                raise RuntimeError(f'Unresolved running record: {path.name}; verify its process first')
        processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
        memory = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
        if processes or any(int(line) > 256 for line in memory.splitlines()):
            raise RuntimeError('GPU occupied; refusing to launch')
        path = root / (args.run_id + '.json')
        record = dict(role='segmentation', executed_by='lcy', run_id=args.run_id, status='running',
                      wrapper_pid=os.getpid(), started_at=datetime.now(timezone.utc).isoformat(),
                      budget_seconds=args.seconds, output=args.output, command=command,
                      git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
        with path.open('x') as stream:
            json.dump(record, stream, indent=2)
        started = time.monotonic()
        process = None
        try:
            process = subprocess.Popen(command, start_new_session=True)
            record['child_pid'] = process.pid
            path.write_text(json.dumps(record, indent=2)+'\n')
            code = process.wait(timeout=args.seconds)
            record.update(status='completed' if code == 0 else 'failed', exit_code=code)
        except BaseException:
            if process is not None and process.poll() is None:
                # Only this wrapper's own process group, never another member's job.
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            record['status'] = 'failed'
            raise
        finally:
            record['elapsed_seconds'] = time.monotonic()-started
            record['finished_at'] = datetime.now(timezone.utc).isoformat()
            path.write_text(json.dumps(record, indent=2)+'\n')
        raise SystemExit(code)


if __name__ == '__main__':
    main()
