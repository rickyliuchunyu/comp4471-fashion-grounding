"""One authorized stage, inside the existing shared lock and GNU hard timeout."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path('/root/autodl-tmp/fashion-grounding')
out = Path(sys.argv[1]).resolve()
repo = Path(__file__).resolve().parents[5]
python = str(root/'envs/segmentation-v1/bin/python')
config = repo/'configs/runs/segmentation'
points = json.loads((config/'s0_resume_20000.json').read_text())['checkpoint_updates']
source = str(root/'vendor/segmentation/Mask2Former')
train = str(root/'shared/processed/fp_train5000_v1')
dev = str(root/'shared/processed/fp_dev250_v1')
commands = [
    ('train', [python, '-m', 'fashion_grounding.segmentation.train', '--package', train,
               '--source', source, '--resume',
               str(root/'runs/segmentation/segmentation-resume5000-train-20261113-r01/checkpoint_005000.pth'),
               '--resume-extension', str(config/'resume_5000_to_20000.json'),
               '--config', str(config/'s0_resume_20000.json'), '--output', str(out/'train')]),
    ('dev', [python, '-m', 'fashion_grounding.segmentation.evaluate', '--package', dev,
             '--training-package', train, '--source', source,
             '--config', str(config/'dev_ap_250_resume_20000.json'), '--output', str(out/'dev'),
             '--checkpoints', *[str(out/'train'/f'checkpoint_{u:06d}.pth') for u in points]]),
    ('new_budget_selection', [python, '-m', 'fashion_grounding.segmentation.select_checkpoint',
                              '--reports', *[str(out/'dev'/f'update_{u:06d}'/'evaluation_report.json') for u in points],
                              '--output', str(out/'new_budget_selection')]),
    ('overall_selection', [python, str(Path(__file__).with_name('compare_overall.py')), str(out)]),
]
os.environ['PYTHONPATH'] = str(repo/'src')
records = []
for name, command in commands:
    record = dict(phase=name, status='running', started_at=datetime.now(timezone.utc).isoformat(),
                  command=command)
    records.append(record)
    (out/'phases.json').write_text(json.dumps(records, indent=2)+'\n')
    print(json.dumps(record), flush=True)
    # Children stay in this process group so the ONE outer timeout covers all phases.
    code = subprocess.call(command, cwd=repo)
    record.update(status='completed' if code == 0 else 'failed', exit_code=code,
                  finished_at=datetime.now(timezone.utc).isoformat())
    (out/'phases.json').write_text(json.dumps(records, indent=2)+'\n')
    if code != 0:
        raise SystemExit(code)
