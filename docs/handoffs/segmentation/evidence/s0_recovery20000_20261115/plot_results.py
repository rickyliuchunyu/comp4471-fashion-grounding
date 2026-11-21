"""Plot saved complete AP reports only; no model or predictions are rerun."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


here = Path(__file__).resolve().parent
repo = here.parents[4]
read = lambda p: json.loads(p.read_text())
old = read(here.parent/'resume5000_20261113/update_003000/evaluation_report.json')
reports = [read(p) for p in sorted((here/'dev').glob('update_*/evaluation_report.json'))]
assert [r['completed_updates'] for r in reports] == list(range(6000, 20001, 2000))
assert all(r['complete'] and r['evaluated_images'] == 250 for r in reports)
selection = read(here/'overall_selection.json')
winner = next(r for r in [old]+reports if r['completed_updates'] == selection['selected']['completed_updates'])
categories = read(repo/'configs/categories.json')['categories']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.titleweight': 'bold'})
fig = plt.figure(figsize=(12, 8), facecolor='white', layout='constrained')
grid = fig.add_gridspec(2, 2, height_ratios=[1.1, 1])
ax = fig.add_subplot(grid[0, :])
updates = [r['completed_updates'] for r in reports]
for key, label, color in [('AP', 'Mask AP', '#145DA0'), ('AP50', 'AP50', '#318875'), ('AP75', 'AP75', '#99623B')]:
    ax.plot(updates, [r['metrics'][key] for r in reports], marker='o', label=label, color=color)
ax.scatter([3000], [old['metrics']['AP']], color='#555555', marker='D', zorder=5, label='Prior best u3000')
ax.axhline(old['metrics']['AP'], color='#777777', linestyle='--', linewidth=1, alpha=.7)
ax.scatter([winner['completed_updates']], [winner['metrics']['AP']], s=180, facecolors='none',
           edgecolors='#b03a48', linewidths=2, zorder=6, label='Selected checkpoint')
for r in reports:
    ax.annotate(f"{r['metrics']['AP']:.2f}", (r['completed_updates'], r['metrics']['AP']),
                xytext=(0, -15), textcoords='offset points', ha='center', fontsize=9, color='#145DA0')
ax.set(title='S0 full dev250 checkpoint comparison', ylabel='COCO mask AP (0-100)', xlabel='Successful optimizer updates',
       xticks=[3000]+updates, ylim=(0, 100))
ax.grid(axis='y', alpha=.18); ax.legend(loc='upper left', ncol=5, fontsize=9)
small = fig.add_subplot(grid[1, 0])
small.plot([3000]+updates, [r['metrics']['AP_small'] for r in [old]+reports], marker='o', color='#7762a3')
small.set(title='Small-object AP', xlabel='Successful optimizer updates', ylabel='AP-small (0-100)')
small.set_ylim(bottom=0); small.grid(axis='y', alpha=.18)
delta = fig.add_subplot(grid[1, 1])
values = [winner['metrics']['per_category_AP'][str(c['id'])]-old['metrics']['per_category_AP'][str(c['id'])]
          for c in categories]
y = np.arange(len(categories))
delta.barh(y, values, color=['#318875' if v >= 0 else '#b03a48' for v in values])
delta.axvline(0, color='#777777', linewidth=.8)
delta.set(yticks=y, yticklabels=[c['name'] for c in categories],
          title=f"Selected u{winner['completed_updates']} minus prior u3000", xlabel='Per-class AP change (points)')
delta.invert_yaxis(); delta.grid(axis='x', alpha=.15)
fig.suptitle('Fashionpedia | eight whole-garment classes | seed 42', fontsize=15)
fig.savefig(here/'s0_dev_comparison.png', dpi=180)
print(here/'s0_dev_comparison.png')
