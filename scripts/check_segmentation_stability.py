"""Hold the outer GPU lock while two fresh trainer processes execute updates 1..200."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from fashion_grounding.data.schema import write_json
from fashion_grounding.segmentation.training_state import read_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('package', 'source', 'checkpoint', 'output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--config', default='configs/runs/segmentation/pilot_stability_200.json')
    args = parser.parse_args()
    config = read_config(args.config)
    if config['purpose'] != 'pilot_200_update_stability' or config['max_updates'] != 200:
        raise ValueError('This coordinator only permits the 200-update stability profile')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    common = [sys.executable, '-m', 'fashion_grounding.segmentation.train', '--package', args.package,
              '--source', args.source, '--config', args.config]
    commands = [common+['--checkpoint', args.checkpoint, '--output', str(output/'updates_001_100'), '--stop-after', '100'],
                common+['--resume', str(output/'updates_001_100/checkpoint_000100.pth'),
                        '--output', str(output/'updates_101_200')]]
    write_json(output/'commands.json', commands)
    for number, command in enumerate(commands, 1):
        with (output/f'phase_{number}.log').open('x') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True,
                           timeout=config['budget_seconds'])
    rows = []
    reports = []
    for directory in ('updates_001_100', 'updates_101_200'):
        rows += [json.loads(line) for line in (output/directory/'metrics.jsonl').read_text().splitlines()]
        reports.append(json.loads((output/directory/'train_report.json').read_text()))
    if [row['update'] for row in rows] != list(range(1, 201)):
        raise ValueError('Updates are missing, repeated or out of order')
    if any(row['optimizer_step_skipped'] for row in rows):
        raise ValueError('AMP skipped an update')
    resume = json.loads((output/'updates_101_200/resume_check.json').read_text())
    if resume['status'] != 'passed' or resume['resumed_from_update'] != 100:
        raise ValueError('Cross-process resume check did not pass')
    losses = np.array([r['total_loss'] for r in rows])
    times = np.array([r['elapsed_seconds'] for r in rows])
    result = dict(status='completed', purpose=config['purpose'], updates=200,
        processes=2, resume_update=100, resume_exact_components=resume['exact_components'],
        all_losses_and_gradients_finite=True, skipped_optimizer_updates=0,
        first_20_mean_loss=float(losses[:20].mean()), last_20_mean_loss=float(losses[-20:].mean()),
        min_loss=float(losses.min()), max_loss=float(losses.max()),
        max_grad_norm_before_clip=max(r['grad_norm_before_clip'] for r in rows),
        max_grad_norm_after_clip=max(r['grad_norm_after_clip'] for r in rows),
        training_update_median_seconds=float(np.median(times)),
        training_update_p95_seconds=float(np.percentile(times, 95)),
        elapsed_process_seconds=sum(r['elapsed_seconds'] for r in reports),
        peak_cuda_allocated_bytes=max(r['peak_cuda_allocated_bytes'] for r in rows),
        peak_cuda_reserved_bytes=max(r['peak_cuda_reserved_bytes'] for r in rows),
        total_images=sum(r['sample_count'] for r in reports),
        empty_augmented_images=sum(r['empty_augmented_images'] for r in reports),
        performance_evaluated=False, final_checkpoint=reports[-1]['checkpoints'][-1],
        git_commit=reports[-1]['git_commit'])
    write_json(output/'stability_report.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
