"""Render recorded stability logs; no model execution or metric estimation."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root, output = Path(args.run), Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    rows = [json.loads(line) for phase in ('updates_001_100', 'updates_101_200')
            for line in (root/phase/'metrics.jsonl').read_text().splitlines()]
    updates = [r['update'] for r in rows]
    if updates != list(range(1, 201)):
        raise ValueError('Expected a complete 200-update trajectory')
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    ax = axes[0, 0]
    losses = [r['total_loss'] for r in rows]
    ax.plot(updates, losses, color='#718bb6', alpha=.55, linewidth=1, label='Per update')
    ax.plot(updates[19:], np.convolve(losses, np.ones(20)/20, mode='valid'), color='#1855a1', linewidth=2, label='Trailing 20 mean')
    ax.set(title='Weighted training loss', ylabel='Loss (log scale)', yscale='log')
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    ax.plot(updates, [r['grad_norm_before_clip'] for r in rows], color='#7a4aa3', label='Before clipping')
    ax.plot(updates, [r['grad_norm_after_clip'] for r in rows], color='#ca652b', label='After clipping')
    ax.set(title='Full-model gradient norm', ylabel='L2 norm (log scale)', yscale='log')
    ax.legend(fontsize=8)
    ax = axes[1, 0]
    for key, label, color in [('cuda_allocated_bytes','Allocated after update','#23846b'),
                              ('cuda_reserved_bytes','Reserved after update','#db9532'),
                              ('peak_cuda_allocated_bytes','Peak allocated so far','#346297')]:
        ax.plot(updates, [r[key]/2**30 for r in rows], label=label, color=color, linewidth=1.4)
    ax.set(title='CUDA memory', ylabel='GiB', ylim=(0, 6))
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    ax.plot(updates, [max(r['learning_rates']) for r in rows], label='Non-backbone', color='#1855a1')
    ax.plot(updates, [min(r['learning_rates']) for r in rows], label='Backbone', color='#777777')
    ax.set(title='Learning rate used by optimizer', ylabel='Learning rate (log scale)', yscale='log')
    ax.legend(fontsize=8)
    for ax in axes.flat:
        ax.axvline(100.5, color='#555555', linestyle='--', linewidth=1)
        ax.set(xlabel='Completed optimizer update', xlim=(1, 200))
        ax.grid(alpha=.2)
    fig.suptitle('200-update pilot stability: LSJ 768, batch 2, AMP scale 1\n20 train images; dashed line = fresh-process resume; no validation evaluation', fontsize=12)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
