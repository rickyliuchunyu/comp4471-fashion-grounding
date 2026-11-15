"""Collect completed-run evidence and derive log summaries without model execution."""
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tarfile
from datetime import datetime, timezone

root = Path('/root/autodl-tmp/fashion-grounding')
run_id = 'segmentation-s0-20000-20261115-r01'
out = root/'runs/segmentation'/run_id
coord_path = root/'coordination'/(run_id+'.json')
coord = json.loads(coord_path.read_text())
if coord['status'] == 'running':
    raise RuntimeError('Refuse final collection while stage is running')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(data)
    return h.hexdigest()


def describe(path):
    return dict(source_path=str(path), sha256=sha(path), bytes=path.stat().st_size)


metrics_path = out/'train/metrics.jsonl'
rows = []
if metrics_path.exists():
    for line in metrics_path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            break  # Retain a hard-stop partial line in the server original.
windows = {}
for start, end in [(5001, 5100)] + [(u-99, u) for u in range(6000, 20001, 2000)]:
    subset = [r for r in rows if start <= r['update'] <= end]
    if not subset:
        continue
    windows[f'{start}_{end}'] = dict(
        rows=len(subset), mean_total_loss=statistics.mean(r['total_loss'] for r in subset),
        mean_loss_components={name: statistics.mean(r['components'][name] for r in subset)
                              for name in ('loss_ce', 'loss_mask', 'loss_dice')},
        mean_auxiliary_losses={kind: statistics.mean(sum(v for k, v in r['components'].items()
                                                       if k.startswith(kind+'_')) for r in subset)
                               for kind in ('loss_ce', 'loss_mask', 'loss_dice')},
        mean_seconds_per_update=statistics.mean(r['elapsed_seconds'] for r in subset),
        learning_rates=sorted({tuple(r['learning_rates']) for r in subset}))
summary = dict(
    derived_from='Existing metrics.jsonl only; no training or evaluator rerun',
    completed_rows=len(rows), first_update=rows[0]['update'] if rows else None,
    last_update=rows[-1]['update'] if rows else None,
    contiguous_updates=[r['update'] for r in rows] == list(range(5001, 5001+len(rows))),
    all_recorded_losses_and_gradient_norms_finite=all(math.isfinite(r[k]) for r in rows for k in
                                                    ('total_loss', 'grad_norm_before_clip', 'grad_norm_after_clip')),
    skipped_optimizer_updates=sum(r['optimizer_step_skipped'] for r in rows),
    windows=windows,
    loss_scales=sorted({r['loss_scale'] for r in rows}),
    sampler_epochs=sorted({r['sampler_epoch'] for r in rows}),
    learning_rates_at_boundaries={str(r['update']): r['learning_rates'] for r in rows
                                  if r['update'] in (5001, 12000, 12001, 16000, 16001, 20000)},
    sample_count=sum(len(r['image_ids']) for r in rows),
    unique_sample_images=len({i for r in rows for i in r['image_ids']}),
    augmented_empty_images=sum(n == 0 for r in rows for n in r['instance_counts']))
if rows:
    summary.update(
        grad_norm_before_clip_min=min(r['grad_norm_before_clip'] for r in rows),
        grad_norm_before_clip_max=max(r['grad_norm_before_clip'] for r in rows),
        grad_norm_after_clip_max=max(r['grad_norm_after_clip'] for r in rows),
        peak_cuda_allocated_bytes=max(r['peak_cuda_allocated_bytes'] for r in rows),
        peak_cuda_reserved_bytes=max(r['peak_cuda_reserved_bytes'] for r in rows))
(out/'metric_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')

postflight = dict(checked_at=datetime.now(timezone.utc).isoformat(),
                  disk={str(p): shutil.disk_usage(p)._asdict() for p in (Path('/'), root)},
                  gpu=subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.used,utilization.gpu',
                                               '--format=csv,noheader'], text=True).strip(),
                  gpu_processes=subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory',
                                                         '--format=csv,noheader'], text=True).strip(),
                  active_coordination_records=[p.name for p in (root/'coordination').glob('*.json')
                                               if read(p).get('status') == 'running'],
                  server_checkout_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                                                cwd=root/'workspaces/segmentation', text=True).strip())
(out/'postflight.json').write_text(json.dumps(postflight, indent=2)+'\n')

files = {}
for name in ('phases.json', 'stage.log', 'metric_summary.json', 'overall_selection.json', 'postflight.json',
             'checkpoint_recovery.json',
             'train/resume_check.json', 'train/budget_extension.json', 'train/resolved_config.json',
             'train/train_report.json', 'train/initialization.json', 'train/data_check.json',
             'train/model_config.yaml', 'train/gradient_failure.json',
             'dev/resolved_config.json', 'dev/data_check.json', 'dev/run_report.json', 'dev/model_config.yaml',
             'new_budget_selection/selection.json'):
    p = out/name
    if p.exists():
        files[name] = p
for p in sorted((out/'dev').glob('update_*/*')):
    if p.name in ('evaluation_report.json', 'evaluation_report.sha256', 'coco_summary.txt'):
        files[str(p.relative_to(out))] = p
files['coordination.json'] = coord_path
meta = read(out/'run_metadata.json')
meta.update(status=coord['status'], finished_at=coord.get('finished_at'),
            elapsed_seconds=coord.get('elapsed_seconds'), coordination=coord,
            phases=read(out/'phases.json'), metric_summary=summary,
            evidence_collection_script=describe(Path(__file__).resolve()),
            evidence_collection_command=[sys.executable, str(Path(__file__).resolve())],
            artifacts={name: describe(p) for name, p in files.items()})
if not meta['status_history'] or meta['status_history'][-1]['status'] != coord['status']:
    meta['status_history'].append(dict(status=coord['status'], at=coord.get('finished_at')))
meta['server_only_artifacts'] = {}
for p in [metrics_path] + sorted((out/'train').glob('checkpoint_*.pth')) + sorted((out/'dev').glob('update_*/predictions.json')):
    if p.exists():
        item = None
        if p.suffix == '.pth' and p.with_suffix('.sha256').exists():
            item = dict(source_path=str(p), bytes=p.stat().st_size,
                        sha256=p.with_suffix('.sha256').read_text().strip(),
                        hash_source='Existing atomic trainer-save sidecar; no additional weight read by this collector')
        elif p.name == 'predictions.json' and (p.parent/'evaluation_report.json').exists():
            item = dict(source_path=str(p), bytes=p.stat().st_size,
                        sha256=read(p.parent/'evaluation_report.json')['predictions_sha256'],
                        hash_source='Existing evaluation report; strict selection verifies this hash')
        meta['server_only_artifacts'][str(p.relative_to(out))] = item or describe(p)
if (out/'overall_selection.json').exists():
    meta['selection'] = read(out/'overall_selection.json')['selected']
(out/'run_metadata.json').write_text(json.dumps(meta, indent=2, allow_nan=False)+'\n')
files['run_metadata.json'] = out/'run_metadata.json'
manifest = {name: describe(p) for name, p in files.items()}
(out/'evidence_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
files['evidence_manifest.json'] = out/'evidence_manifest.json'
archive = out/'evidence.tar.gz'
with tarfile.open(archive, 'w:gz') as tar:
    for name, path in files.items():
        tar.add(path, arcname=name, recursive=False)
print(json.dumps(dict(status=meta['status'], archive=describe(archive), files=len(files),
                      selection=meta.get('selection'), completed_updates=summary['last_update'],
                      elapsed_seconds=meta.get('elapsed_seconds')), indent=2))
