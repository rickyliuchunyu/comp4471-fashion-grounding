"""Publish immutable server aliases after actual consumer verification; no inference."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from fashion_grounding.data.schema import read_json, write_json, sha256, require

root = Path('/root/autodl-tmp/fashion-grounding')
run = root/'runs/segmentation/segmentation-u3000-dev-candidates-20261114-r01'
shared = root/'shared/candidates/u3000-dev250-20261114-r01'
receipt = read_json(run/'offline_receipt.json')
coordination = root/'coordination/segmentation-u3000-dev-candidates-20261114-r01.json'
gpu = read_json(coordination)
require(gpu['status'] == 'completed' and gpu['exit_code'] == 0, 'GPU stage not completed')
require(receipt['status'] == 'completed' and len(receipt['packages']) == 3, 'Offline stage not completed')
shared.mkdir(exist_ok=False)
links = {'cache': run/'cache', **{f'trial-{p["threshold"]}': Path(p['path']) for p in receipt['packages']}}
for name, target in links.items():
    (shared/name).symlink_to(target, target_is_directory=True)
    require((shared/name).resolve() == target.resolve(), 'Delivery alias points elsewhere')
for p in receipt['packages']:
    require(sha256(shared/f'trial-{p["threshold"]}'/'candidates_predicted.json') ==
            p['candidate_manifest_sha256'], 'Published manifest differs')
write_json(shared/'delivery_manifest.json', dict(status='available',
    published_at=datetime.now(timezone.utc).isoformat(), server_aliases=True,
    storage_policy='Absolute symlinks to immutable run directories; preserve source directories. Each resolved package is self-contained.',
    source_cache_sha256=receipt['source_cache_sha256'], selected_threshold=None,
    consumer_verification='lcy executed pinned wyz 103dcc5 reader on all actual outputs; teammate sign-off pending',
    packages=[dict(threshold=p['threshold'], path=str(shared/f'trial-{p["threshold"]}'),
                   manifest_sha256=p['candidate_manifest_sha256'], images=p['images'],
                   candidates=p['candidates'], no_candidates_images=p['no_candidates_images'])
              for p in receipt['packages']]))
postflight = dict(
    disk=subprocess.check_output(['df', '-h', '/', str(root)], text=True),
    gpu=subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu', '--format=csv,noheader'], text=True),
    compute_processes=subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'], text=True),
    running_records=[p.name for p in (root/'coordination').glob('*.json') if read_json(p).get('status') == 'running'])
write_json(run/'delivery_receipt.json', dict(status='completed', shared_directory=str(shared),
    delivery_manifest_sha256=sha256(shared/'delivery_manifest.json'),
    source_cache_sha256=receipt['source_cache_sha256'],
    gpu_elapsed_seconds=gpu['elapsed_seconds'], gpu_hard_limit_seconds=gpu['budget_seconds'],
    offline_elapsed_seconds=receipt['elapsed_seconds'], offline_hard_limit_seconds=1800,
    postflight=postflight))
print(json.dumps(read_json(run/'delivery_receipt.json'), indent=2))
