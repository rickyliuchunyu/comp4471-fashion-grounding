"""Pin only new query/config metadata; no pixels, masks, checkpoint load or GPU."""
from collections import Counter
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys

from fashion_grounding.data.schema import read_json, write_json, sha256, require
from fashion_grounding.segmentation.export_candidates import validate_config, input_manifest
from fashion_grounding.segmentation.select_threshold import load_frozen_dev


ROOT = Path('/root/autodl-tmp/fashion-grounding')
REPO = ROOT/'workspaces/segmentation'
RUN_ID = 'segmentation-u50000-dev-candidates-20261118-r01'


def main():
    dest = ROOT/'runs/segmentation'/f'{RUN_ID}-inputs'
    require(not dest.exists(), 'Immutable input directory already exists')
    config = validate_config(read_json(REPO/'configs/runs/segmentation/candidate_u50000_export.json'))
    final = read_json(REPO/'docs/handoffs/segmentation/evidence/tenk_completion_20261118/final_checkpoint.json')
    require(config['checkpoint_sha256'] == final['checkpoint_sha256'] and config['completed_updates'] == 50000,
            'Wrong selected checkpoint')
    checkpoint = Path(final['checkpoint_path'])
    require(checkpoint.stat().st_size == final['checkpoint_bytes'] and
            checkpoint.with_suffix('.sha256').read_text().strip() == final['checkpoint_sha256'],
            'Checkpoint/sidecar identity differs; full hash checked during required inference load')
    dev = ROOT/'shared/processed/fp_dev250_v1'
    manifest, images = input_manifest(dev, config, 'dev')
    language = ROOT/'shared/processed/fp_language_dev250_reviewed_v1'
    query_hash = 'ac8d6af9d1b854452627b2f6d72aed58a91bee0642344ac37932f8701fb841d9'
    require(sha256(language/'queries_accepted.jsonl') == query_hash, 'Published query version changed')
    require(sha256(language/'annotations_coco.json') == sha256(dev/'annotations_coco.json') ==
            final['evaluation_data']['annotation_sha256'], 'Language/segmentation GT differs')
    lm = {im['image_id']: im for im in read_json(language/'image_manifest.json')['images']}
    require(set(lm) == set(images), 'Language/dev image inventory differs')
    for iid, im in images.items():
        require(all(im[k] == lm[iid][k] for k in ('sha256', 'width', 'height', 'split', 'group_id')),
                'Language/dev image identity differs')
    queries = [json.loads(line) for line in (language/'queries_accepted.jsonl').read_text().splitlines() if line.strip()]
    require(len(queries) == 642 and len({q['image_id'] for q in queries}) == 246 and
            len({q['semantic_query_id'] for q in queries}) == 321 and
            all(q['review_status'] == 'accepted' for q in queries), 'Wrong reviewed query subset')
    require(sha256(language/'human_review.json') ==
            'fdbd519b6d5851d5c2514798efae6b18a9133db02cef6731cc6c4c58dc8c9ab6', 'Review evidence changed')
    require(sha256(language/'review_release.json') ==
            '19d1a4ec0b4907add07202b959c89c8029c2581306f15a23bb05acb6a0d629fb', 'Release evidence changed')
    consumer = ROOT/'runs/segmentation/segmentation-u18000-dev-candidates-20261115-r01/grounding_inputs.py'
    require(sha256(consumer) == '74153d2e43d34379d9f6b9b0c9537742402010b00c140935ada7191d03cb13a8',
            'Pinned wyz reader changed')
    active = [p.name for p in (ROOT/'coordination').glob('*.json') if read_json(p).get('status') == 'running']
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader'], text=True).strip()
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    with (ROOT/'coordination/gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not active and not processes, 'GPU/coordination occupied')
    require(shutil.disk_usage(ROOT).free >= 3*1024**3, 'Less than 3 GiB free')
    dest.mkdir()
    for name in ('queries_accepted.jsonl', 'human_review.json', 'review_release.json', 'review_audit.json'):
        shutil.copy2(language/name, dest/name)
    shutil.copy2(consumer, dest/'grounding_inputs.py')
    now = datetime.now(timezone.utc).isoformat()
    authorization = dict(schema_version=1, authority='current_user',
        authorized_use='offline_candidate_threshold_selection', confirmed_by='current_user', confirmed_at='2026-11-18',
        queries_sha256=query_hash, recorded_at=now,
        user_statement='那就先核对整组 GitHub、服务器磁盘和 GPU，用 50k 最佳权重对完整 dev250 做一次候选推理，保存真实 query ID 和阈值前缓存，再离线比较 0.3/0.5/0.7 并选择固定候选版本。复用已有接口和检查，只补新产物的必要验证，不重复 smoke、不追加训练，只做我的分割部分，设置硬时间上限，完成后同步 GitHub。',
        note='User authorizes development selection on the current published dev; no new claim of full-package human acceptance or formal team freeze.')
    write_json(dest/'authorization.json', authorization)
    ref = lambda path: dict(path=str(path), sha256=sha256(path))
    freeze = dict(schema_version=1, purpose='user_authorized_language_dev', split='dev',
        development_freeze=True, formal_freeze=False, confirmed_by='current_user', confirmed_at='2026-11-18',
        authorization_record=ref(dest/'authorization.json'), queries=ref(dest/'queries_accepted.jsonl'),
        image_manifest=ref(manifest), annotations=ref(dev/'annotations_coco.json'),
        review_records=[ref(dest/n) for n in ('human_review.json', 'review_release.json', 'review_audit.json')],
        source_query=ref(language/'queries_accepted.jsonl'),
        scope=dict(kind='user_authorized_dev_subset', query_count=642,
            query_image_ids=sorted({q['image_id'] for q in queries}),
            description='Current wkq reviewed dev: 321 bilingual semantic groups / 642 accepted records on 246 of dev250 images; pending excluded by upstream before prediction.'),
        note='Preserve accepted statuses and review provenance. Upstream package_complete_human_acceptance=false; consumer development freeze is not a full-package or final-test acceptance.')
    write_json(dest/'freeze.json', freeze)
    load_frozen_dev(dest/'freeze.json', final['evaluation_data'], allow_user_authorized_dev=True)
    record = dict(status='passed', checked_at=now, gpu=gpu, gpu_processes=processes,
        active_coordination=active, gpu_lock_available=True, free_data_bytes=shutil.disk_usage(ROOT).free,
        new_query_metadata_checked=True, query_records=642, semantic_groups=321, query_images=246,
        query_review_status_counts=dict(Counter(q['review_status'] for q in queries)),
        queries_sha256=query_hash, freeze_sha256=sha256(dest/'freeze.json'),
        image_manifest_sha256=sha256(manifest), annotation_sha256=sha256(dev/'annotations_coco.json'),
        source_language_manifest_sha256=sha256(language/'image_manifest.json'),
        input_manifest_identity_equal=True, checkpoint_sidecar_matches_selected=True,
        decoder_sha256=sha256(REPO/'src/fashion_grounding/segmentation/predict.py'),
        consumer_sha256=sha256(dest/'grounding_inputs.py'),
        model_loads=0, inference_calls=0, optimizer_updates=0, smoke_reruns=0,
        reused='Existing dev250/10k acceptance, model/head/stability/resume, decoder, matcher and consumer implementation evidence.')
    require('torch' not in sys.modules, 'Preparation unexpectedly imported torch')
    write_json(dest/'server_preflight.json', record)
    print(json.dumps(record))


if __name__ == '__main__':
    main()
