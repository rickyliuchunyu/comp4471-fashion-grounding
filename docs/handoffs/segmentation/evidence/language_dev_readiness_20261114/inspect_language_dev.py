"""Read published language readiness only; no pixels, masks, GT matching or model."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

root = Path('/root/autodl-tmp/fashion-grounding/shared/processed')


def inspect(path):
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw),
        rows=len(rows), images=len({r['image_id'] for r in rows}),
        split_counts=dict(Counter(r.get('split') for r in rows)),
        review_status_counts=dict(Counter(r.get('review_status') for r in rows)),
        accepted_dev_rows=sum(r.get('split') == 'dev' and r.get('review_status') == 'accepted' for r in rows))


accepted = [inspect(p) for p in sorted(root.glob('*/queries_accepted.jsonl'))]
dev = root/'fp_language_dev_draft_v2'
draft = inspect(dev/'queries_draft.jsonl')
release_path = dev/'RELEASE.json'
release_raw = release_path.read_bytes()
release = json.loads(release_raw)
manifest = json.loads((dev/'image_manifest.json').read_text())
output = dict(checked_at_utc=datetime.now(timezone.utc).isoformat(),
    scope='Published shared/processed package inventory, accepted JSONL counts and current language-dev metadata only',
    published_packages=sorted(p.name for p in root.iterdir() if p.is_dir()),
    accepted_files=accepted, dev_draft=draft,
    dev_manifest_images=len(manifest['images']),
    dev_manifest_sha256=hashlib.sha256((dev/'image_manifest.json').read_bytes()).hexdigest(),
    release=dict(path=str(release_path), sha256=hashlib.sha256(release_raw).hexdigest(), contents=release),
    accepted_dev_rows_across_published_accepted_files=sum(f['accepted_dev_rows'] for f in accepted),
    selected_threshold=None, matched_query_coverage=None,
    model_inference_calls=0, checkpoint_loads=0, optimizer_updates=0, candidate_revalidation_runs=0)
output['readiness'] = 'not_ready_no_accepted_dev' if not output['accepted_dev_rows_across_published_accepted_files'] else 'requires_freeze_and_query_validation'
print(json.dumps(output, ensure_ascii=False, indent=2))
