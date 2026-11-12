"""Consume new train/dev inputs on CPU; reuse prior model validation."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from pathlib import Path
from collections import Counter
import json
import subprocess
import sys
import time

BASE = Path('/root/autodl-tmp/fashion-grounding')
WORK = BASE/'workspaces/segmentation'
ROOT = BASE/'shared/processed'
OUT = BASE/'runs/segmentation/large-input-acceptance-20261112-r01'
sys.path.insert(0, str(WORK/'src'))
from fashion_grounding.data.schema import read_json, sha256, safe_path, require, write_json
from fashion_grounding.segmentation.data import load_training_package
from fashion_grounding.segmentation.evaluation import load_eval_package

PINS = {
    'fp_train5000_v1': {
        'image_manifest.json': '82916717a867c1e2fe4ae8b42ea34072e97b3edef9113ecf83a855e390a0c0f3',
        'SHA256SUMS.json': '92f57e8fe39191bcc80495bfbbda20f3359aa9a0ce42d85c51791f876dd6f953'},
    'fp_dev250_v1': {
        'image_manifest.json': 'c8c3b24d5a21f93a32060d2fdd9772102bf48b0d6504d60a3466444c7866ecdc',
        'SHA256SUMS.json': 'a051bb8808b3ac484da54b525218bc52e86c86ea6c16f11a05863ec16d0893b0'}
}

def main():
    started = time.monotonic()
    require(not OUT.exists(), 'Do not overwrite a run')
    OUT.mkdir(parents=True)
    source_hashes = {name: sha256(WORK/name) for name in (
        'src/fashion_grounding/data/schema.py',
        'src/fashion_grounding/segmentation/data.py',
        'src/fashion_grounding/segmentation/evaluation.py')}
    release = read_json(ROOT/'fp_data_eval_large_v1/release_index.json')
    metadata = {}
    for name, pins in PINS.items():
        package = ROOT/name
        for filename, expected in pins.items():
            require(sha256(package/filename) == expected, 'Release pin differs: '+name+'/'+filename)
        checksums = read_json(package/'SHA256SUMS.json')
        verified = {}
        # Only segmentation consumer inputs and provenance; producer already
        # verified per-candidate files and source-mask equality.
        for filename in ('image_manifest.json', 'annotations_coco.json', 'mapping_config.json', 'RELEASE.json'):
            digest = sha256(package/filename)
            require(checksums[filename] == digest, 'Consumer metadata hash mismatch')
            verified[filename] = digest
        manifest = read_json(package/'image_manifest.json')
        coco = read_json(package/'annotations_coco.json')
        expected_split = 'train' if name == 'fp_train5000_v1' else 'dev'
        source_split = 'official_train2020' if expected_split == 'train' else 'official_val2020'
        offset = 0 if expected_split == 'train' else 10**9
        require(coco['schema_version'] == 3 and coco['illustrative_only'] is False, 'Real COCO schema required')
        require(all(im['source_split'] == source_split and im['split'] == expected_split for im in manifest['images']), 'Official split mismatch')
        require(all(im['image_id'] == int(im['source_image_id'])+1+offset for im in manifest['images']), 'Image namespace mismatch')
        require(all(a['id'] == int(a['source_annotation_id'])+1+offset for a in coco['annotations']), 'Annotation namespace mismatch')
        for im in manifest['images']:
            require(checksums[im['file_name']] == im['sha256'], 'Image pin differs between manifests')
        require(len(manifest['images']) == release['packages'][name]['images'], 'Release image count differs')
        require(len(coco['annotations']) == release['packages'][name]['instances'], 'Release instance count differs')
        metadata[name] = dict(verified_metadata_sha256=verified, release=read_json(package/'RELEASE.json'),
            images=len(manifest['images']), instances=len(coco['annotations']), source_split=source_split,
            project_category_counts=dict(Counter(str(a['category_id']) for a in coco['annotations'])),
            checksum_manifest_sha256=pins['SHA256SUMS.json'], namespace_offset=offset,
            image_id_range=[min(im['image_id'] for im in manifest['images']), max(im['image_id'] for im in manifest['images'])])
    print('Pinned consumer metadata, official split and ID namespaces: PASS', flush=True)
    records, train = load_training_package(ROOT/'fp_train5000_v1')
    require(len(records) == 5000 and train['instances'] == 17775, 'Unexpected training input counts')
    require({a['category_id'] for r in records for a in r['annotations']} == set(range(8)), 'Internal mapping mismatch')
    print('Existing train reader: 5000 RGB images / 17775 instances / internal classes 0..7 PASS', flush=True)
    dev_config = dict(purpose='dev_checkpoint_selection',
        image_manifest_sha256=PINS['fp_dev250_v1']['image_manifest.json'],
        annotation_sha256=metadata['fp_dev250_v1']['verified_metadata_sha256']['annotations_coco.json'])
    _, _, dev = load_eval_package(ROOT/'fp_dev250_v1', dev_config, ROOT/'fp_train5000_v1')
    require(dev['image_count'] == 250 and dev['annotation_count'] == 1008, 'Unexpected dev input counts')
    print('Existing dev reader: 250 RGB images / 1008 instances; train/dev overlap zero PASS', flush=True)
    for name, info in metadata.items():
        for filename, digest in info['verified_metadata_sha256'].items():
            require(sha256(ROOT/name/filename) == digest, 'Consumer metadata changed during read')
    require(all(sha256(WORK/name) == digest for name, digest in source_hashes.items()), 'Reader source changed')
    result = dict(status='passed', scope='New segmentation input compatibility only; not formal freeze or model validation',
        reader_git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=WORK,text=True).strip(),
        reader_source_sha256=source_hashes, script_sha256=sha256(__file__),
        release_index_sha256=sha256(ROOT/'fp_data_eval_large_v1/release_index.json'),
        upstream_code_commit=release['code_commit'], upstream_formal_freeze=release['formal_freeze'],
        package_metadata=metadata, train_reader=train, dev_reader=dev,
        consumer_images_hashed_and_decoded=5250, consumer_coco_instances_geometry_checked=18783,
        gpu_used=False, model_loaded=False, optimizer_updates=0, checkpoint_operations=0,
        final_test_opened=False, language_queries_read=False,
        reused_evidence=['COCO weights/head verification', '200-update stability and process resume', 'AP/selection implementation validation', 'upstream source-mask equality and candidate file validation'],
        not_repeated=['source-mask equality', 'all producer checksums', 'model forward/backward', 'checkpoint save/restore', 'full test suites'],
        formal_selection_ready=False, remaining=['team segmentation split acceptance/freeze and near-duplicate review', 'checkpoints trained on the matching frozen train inputs'],
        elapsed_seconds=time.monotonic()-started)
    write_json(OUT/'acceptance.json', result)
    print(json.dumps(dict(status='passed', report=str(OUT/'acceptance.json'), seconds=result['elapsed_seconds'], gpu_used=False)), flush=True)

if __name__ == '__main__':
    main()
