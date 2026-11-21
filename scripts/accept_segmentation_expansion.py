"""Incremental CPU consumer acceptance of wkq's pinned targeted 10k release.

No model, mapper, optimizer, inference, or training. Reuse producer source-pixel
and near-duplicate audits; decode only the added images/masks with our real reader.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

from fashion_grounding.data.schema import read_json, require, safe_path, sha256, write_json


def indexed(rows, key):
    result = {r[key]: r for r in rows}
    require(len(result) == len(rows), f'duplicate {key}')
    return result


def check_subsets(parent, merged, old_ids, added_ids, key):
    old = indexed(parent, key)
    full = indexed(merged, key)
    require(len(old_ids) == len(set(old_ids)), 'duplicate old membership')
    require(len(added_ids) == len(set(added_ids)), 'duplicate added membership')
    require(set(old) == set(old_ids), 'old membership differs from parent')
    require(not set(old_ids) & set(added_ids), 'old/added overlap')
    require(set(old_ids) | set(added_ids) == set(full), 'membership does not cover release')
    require(all(full[k] == v for k, v in old.items()), 'old record changed')
    added_set = set(added_ids)
    return [r for r in merged if r[key] in added_set]


def identity_sets(rows):
    return {
        'image_id': {r['image_id'] for r in rows},
        'source_identity': {(r['source_dataset'], r['source_split'], str(r['source_image_id'])) for r in rows},
        'group_id': {r['group_id'] for r in rows},
        'sha256': {r['sha256'] for r in rows},
    }


def check_isolation(added, protected):
    new = identity_sets(added)
    require(all(len(v) == len(added) for v in new.values()), 'duplicate added identity/group/hash')
    overlaps = {}
    for name, rows in protected.items():
        other = identity_sets(rows)
        overlaps[name] = {key: len(values & other[key]) for key, values in new.items()}
    require(not any(n for counts in overlaps.values() for n in counts.values()), 'cross-partition overlap')
    return overlaps


def accept(plan, output):
    started = time.monotonic()
    root = Path(plan['package'])
    parent = Path(plan['parent'])
    pins = plan['package_pins']
    for name, digest in pins.items():
        require(sha256(root / name) == digest, f'release pin differs: {name}')
    catalog = read_json(root / 'SHA256SUMS.json')
    evidence_hashes = {}

    def released(name):
        path = safe_path(root, name)
        digest = sha256(path)
        require(digest == catalog[name], f'release catalog differs: {name}')
        evidence_hashes[name] = digest
        return read_json(path)

    manifest, coco = released('image_manifest.json'), released('annotations_coco.json')
    release = released('RELEASE.json')
    require(release['release_id'] == root.name and release['parent'] == parent.name, 'wrong lineage')
    require(release['code_commit'] == plan['producer_build_commit'], 'producer revision differs')
    protected, frozen_catalogs = {}, {}
    for name, package in [('parent', parent), *[(k, Path(v)) for k, v in plan['protected'].items()]]:
        before = release['frozen_inputs'][package.name]
        require(sha256(package/'SHA256SUMS.json') == before['checksums_file'], 'protected catalog changed')
        current = read_json(package/'SHA256SUMS.json')
        require(current == before['files'], 'protected file catalog changed')
        require(sha256(package/'image_manifest.json') == plan['protected_manifest_pins'][name], 'protected manifest changed')
        require(current['image_manifest.json'] == plan['protected_manifest_pins'][name], 'protected manifest pin differs')
        protected[name] = read_json(package/'image_manifest.json')['images']
        frozen_catalogs[name] = current
    require(sha256(parent/'annotations_coco.json') == plan['parent_annotation_sha256'], 'parent annotations changed')
    old_coco = read_json(parent/'annotations_coco.json')
    old_members, new_members = released('old_subset_membership.json'), released('added_subset_membership.json')
    added = check_subsets(protected['parent'], manifest['images'], old_members['image_ids'], new_members['image_ids'], 'image_id')
    added_images = check_subsets(old_coco['images'], coco['images'], old_members['image_ids'], new_members['image_ids'], 'id')
    added_anns = check_subsets(old_coco['annotations'], coco['annotations'], old_members['annotation_ids'], new_members['annotation_ids'], 'id')
    require(len(manifest['images']) == 10000 and len(added) == 5000 and len(added_anns) == 21924, 'unexpected counts')
    require(manifest['schema_version'] == coco['schema_version'] == 3, 'wrong schema')
    require(manifest['illustrative_only'] is False and coco['illustrative_only'] is False, 'illustrative package')
    require(coco['categories'] == old_coco['categories'], 'category mapping changed')
    require(sorted(c['id'] for c in coco['categories']) == list(range(1, 9)), 'expected eight classes')
    require(all(im['split'] == 'train' and im['source_dataset'] == 'fashionpedia'
                and im['source_split'] == 'official_train2020'
                and im['image_id'] == int(im['source_image_id']) + 1 for im in manifest['images']), 'non-train source/ID')
    ids = set(new_members['image_ids'])
    require(all(a['image_id'] in ids and a['id'] == a['source_annotation_id'] + 1
                and a['source_dataset'] == 'fashionpedia' for a in added_anns), 'added annotation lineage differs')
    mapping = released('mapping_config.json')
    require(sha256(root/'mapping_config.json') == frozen_catalogs['parent']['mapping_config.json'], 'mapping changed')
    source_map = {x['source_id']: x['project_category_id'] for x in mapping['fashionpedia'] if x['action'] == 'map'}
    require(all(source_map.get(a['source_category_id']) == a['category_id'] for a in added_anns), 'source category mapping differs')
    old_assets = {k: v for k, v in frozen_catalogs['parent'].items() if k.startswith(('images/', 'masks/'))}
    require(all(catalog.get(k) == v for k, v in old_assets.items()), 'old asset catalog differs')
    overlaps = check_isolation(added, protected)

    # Frozen dev/test contents are not revalidated: only their manifest identities
    # are used for new-image isolation. No test masks, annotations or images read.
    sidecar = released('sampling_sidecar.json')
    side = indexed(sidecar, 'image_id')
    require(set(side) == ids, 'sidecar membership differs')
    require(all(str(side[x['image_id']]['source_image_id']) == x['source_image_id']
                and side[x['image_id']]['group_id'] == x['group_id']
                and side[x['image_id']]['sha256'] == x['sha256'] for x in added), 'sidecar identity differs')
    sampling = dict(Counter(x['sampling'] for x in sidecar))
    require(sampling == {'broad': 3500, 'targeted': 1500}, 'sampling ratio differs')
    duplicate = released('audit/duplicate_audit.json')
    independent = released('audit/independent_duplicate_check.json')
    require(independent['threshold'] == 6 and independent['flags'] == [], 'producer duplicate flags')
    require(independent['comparisons'] == 'all four original/mirror orientations; new vs frozen and new vs new', 'wrong duplicate audit scope')
    quarantined = {x['source_image_id'] for x in duplicate['quarantine']}
    require(not quarantined & {x['source_image_id'] for x in sidecar}, 'quarantined image selected')
    preservation = released('preservation_report.json')
    source_pixels = released('audit/added_consumer_check_report.json')
    require(preservation['status'] == source_pixels['status'] == 'passed', 'producer preservation/source check failed')
    require(preservation['old_json_records_equal'] is True and preservation['new_duplicate_flags'] == 0, 'producer retention failed')
    require(preservation['old_assets_identical_in_merged'] == len(old_assets), 'asset audit count differs')
    require(source_pixels['source_masks_pixel_identical'] == len(added_anns), 'source pixel coverage differs')
    visual = released('audit/visual_audit.json')
    distribution = released('audit/distribution_report.json')

    print('metadata, old subset, new identities and audit bindings passed; reading additions', flush=True)
    # Hard links preserve our reader's path confinement without copying images or
    # weakening safe_path. The temporary view is read-only by use; never published.
    from fashion_grounding.segmentation.data import load_training_package
    with tempfile.TemporaryDirectory(prefix='added-view-', dir=output) as directory:
        view = Path(directory)
        write_json(view/'image_manifest.json', {**manifest, 'images': added})
        write_json(view/'annotations_coco.json', {**coco, 'images': added_images, 'annotations': added_anns})
        for im in added:
            target = view / im['file_name']
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(safe_path(root, im['file_name']), target)
        records, loader = load_training_package(view)
        require({r['image_id'] for r in records} == ids, 'consumer returned different images')
        require(loader['instances'] + loader['crowd_skipped'] == len(added_anns), 'consumer lost annotations')
        del records
    report = {
        'status': 'passed', 'acceptance_scope': 'segmentation_input_only',
        'checked_at': datetime.now(timezone.utc).isoformat(), 'elapsed_seconds': time.monotonic()-started,
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'script_sha256': sha256(__file__), 'loader_sha256': sha256(Path(__file__).resolve().parents[1]/'src/fashion_grounding/segmentation/data.py'),
        'package': str(root), 'package_pins': pins, 'producer_build_commit': release['code_commit'],
        'protected_manifest_pins': plan['protected_manifest_pins'],
        'total_images': len(manifest['images']), 'total_instances': len(coco['annotations']),
        'total_groups': len({x['group_id'] for x in manifest['images']}),
        'old_records_exact': {'images': len(protected['parent']), 'annotations': len(old_coco['annotations'])},
        'old_asset_catalog_matches': len(old_assets), 'old_asset_bytes': 'reused pinned producer preservation audit; not reread',
        'added_loader': loader, 'added_cross_partition_overlap_counts': overlaps,
        'sampling': sampling, 'quarantined_not_selected': len(quarantined),
        'reused_audits': {'source_pixels': source_pixels['source_masks_pixel_identical'], 'near_duplicates': independent,
                          'old_asset_bytes': preservation['old_assets_identical_in_merged']},
        'evidence_hashes': evidence_hashes,
        'distribution': {k: {name: distribution[k][name] for name in ['images', 'groups', 'instances', 'small_instances_le1024', 'extreme_instances_le256']} for k in ['old', 'added', 'combined']},
        'soft_goals': distribution['soft_goals'], 'visual_review_provenance': visual,
        'formal_three_party_freeze': False, 'test_access': 'manifest and checksum catalog only',
        'model_loads': 0, 'inference_images': 0, 'training_updates': 0, 'smoke_runs': 0,
        'limitations': ['No performance claim; targeted expansion changes size and composition.',
                       'Reused pHash audit is conservative evidence, not proof of all identity/crop relationships.',
                       'Visual audit is Codex spot checking, not human exhaustive certification.'],
    }
    write_json(output/'acceptance.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Run with CUDA_VISIBLE_DEVICES empty')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    plan = read_json(args.plan)
    write_json(output/'acceptance_plan.json', plan)
    try:
        result = accept(plan, output)
    except Exception as error:
        write_json(output/'failure.json', {'status': 'failed', 'error': str(error)})
        raise
    print(json.dumps({k: result[k] for k in ['status', 'elapsed_seconds', 'total_images', 'total_instances']}, indent=2))


if __name__ == '__main__':
    main()
