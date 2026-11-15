"""One authorized recovery/AP stage; run inside shared lock and one hard timeout."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--plan', required=True)
    args = p.parse_args()
    repo = Path(__file__).resolve().parents[1]
    plan_path = Path(args.plan).resolve()
    plan = json.loads(plan_path.read_text())
    out = Path(args.output).resolve()
    if (out/'phases.json').exists():
        raise FileExistsError('Recovery stage already attempted; no automatic restart')
    root = Path('/root/autodl-tmp/fashion-grounding')
    train, dev = (str(root/'shared/processed'/name) for name in ('fp_train5000_v1', 'fp_dev250_v1'))
    source = str(root/'vendor/segmentation/Mask2Former')
    checkpoints = [item['path'] for item in plan['existing_checkpoints']] + [str(out/'train/checkpoint_020000.pth')]
    commands = [
        ('amp_cuda_regression', [sys.executable, 'tests/test_segmentation_amp_recovery.py',
            'AmpRecoveryTests.test_actual_scaler_skips_optimizer_and_replays_same_rng_sampler_buffers', '-v']),
        ('train', [sys.executable, '-m', 'fashion_grounding.segmentation.train', '--package', train,
            '--source', source, '--resume', plan['parent_checkpoint_path'], '--resume-repair',
            str(repo/plan['repair_policy']), '--config', str(repo/plan['training_config']), '--output', str(out/'train')]),
        ('dev', [sys.executable, '-m', 'fashion_grounding.segmentation.evaluate', '--package', dev,
            '--training-package', train, '--source', source, '--config', str(repo/plan['evaluation_config']),
            '--output', str(out/'dev'), '--checkpoints', *checkpoints]),
        ('overall_selection', [sys.executable, 'scripts/select_s0_recovery.py', '--output', str(out),
            '--plan', str(plan_path)]),
    ]
    env = dict(os.environ, PYTHONPATH=str(repo/'src'), AMP_TEST_DEVICE='cuda')
    records = []
    for name, command in commands:
        record = dict(phase=name, status='running', command=command,
                      started_at=datetime.now(timezone.utc).isoformat())
        records.append(record)
        (out/'phases.json').write_text(json.dumps(records, indent=2)+'\n')
        print(json.dumps(record), flush=True)
        # No new session: every child remains in the outer timeout's process group.
        code = subprocess.call(command, cwd=repo, env=env)
        record.update(status='completed' if code == 0 else 'failed', exit_code=code,
                      finished_at=datetime.now(timezone.utc).isoformat())
        (out/'phases.json').write_text(json.dumps(records, indent=2)+'\n')
        if code:
            raise SystemExit(code)


if __name__ == '__main__': main()
