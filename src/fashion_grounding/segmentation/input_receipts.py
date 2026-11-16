"""Compose immutable dev validation with an independently accepted train expansion.

This verifies metadata identities and receipts only. It never opens image files,
decodes masks, creates a model, or changes the historical acceptance records.
"""
from pathlib import Path

from ..data.schema import read_json, require, sha256
from .evaluation import PROTOCOL, reuse_eval_package


def read_evidence_bundle(path):
    path = Path(path).resolve()
    bundle = read_json(path)
    require(bundle['schema_version'] == 1, 'Unknown validation bundle schema')
    require(bundle['mode'] in ('unchanged_train', 'accepted_expansion'), 'Unknown reuse mode')
    required = {'dev_report'}
    if bundle['mode'] == 'accepted_expansion':
        required |= {'acceptance', 'acceptance_plan', 'acceptance_run'}
    require(set(bundle['references']) == required, 'Incomplete evidence references')
    records, paths = {}, {}
    for name, ref in bundle['references'].items():
        target = (path.parent/ref['path']).resolve()
        require(sha256(target) == ref['sha256'], 'Evidence hash changed: '+name)
        records[name], paths[name] = read_json(target), target
    return bundle, records, paths


def zero_counts(counts, keys):
    require(set(counts) == set(keys) and all(type(v) is int and v == 0 for v in counts.values()),
            'Missing or nonzero partition evidence')


def validate_chain(bundle, records):
    report = records['dev_report']
    require(report['purpose'] == 'dev_checkpoint_selection' and report['protocol'] == PROTOCOL
            and report['complete'] is True and report['selection_eligible'] is True
            and report['inference_errors'] == 0, 'Need completed dev receipt')
    zero_counts(report['overlap'], ('image_id', 'sha256', 'group_id', 'source_image_id'))
    require(report['evaluation_data'] == bundle['evaluation_data'], 'Dev receipt identity differs')
    require(report['evaluated_images'] == bundle['dev_images'], 'Dev receipt population differs')
    if bundle['mode'] == 'unchanged_train':
        require(report['training_data'] == bundle['training_data'], 'Training identity changed without expansion')
        return
    receipt, plan, run = (records[k] for k in ('acceptance', 'acceptance_plan', 'acceptance_run'))
    require(receipt['status'] == 'passed' and receipt['acceptance_scope'] == 'segmentation_input_only'
            and run['status'] == 'completed' and run['exit_code'] == 0, 'Incomplete expansion acceptance')
    for key, artifact in [('acceptance', 'acceptance.json'), ('acceptance_plan', 'acceptance_plan.json')]:
        require(run['artifacts'][artifact]['sha256'] == bundle['references'][key]['sha256'],
                'Acceptance run does not bind '+key)
    require(run['git_commit'] == receipt['source_commit'], 'Acceptance source differs')
    require(receipt['package_pins'] == plan['package_pins']
            and receipt['protected_manifest_pins'] == plan['protected_manifest_pins']
            and receipt['producer_build_commit'] == plan['producer_build_commit'], 'Acceptance plan differs')
    parent = dict(image_manifest_sha256=plan['protected_manifest_pins']['parent'],
                  annotation_sha256=plan['parent_annotation_sha256'])
    expanded = dict(image_manifest_sha256=receipt['package_pins']['image_manifest.json'],
                    annotation_sha256=receipt['package_pins']['annotations_coco.json'])
    require(parent == report['training_data'], 'Expansion parent differs from prior dev receipt')
    require(expanded == bundle['training_data'], 'Expanded training identity differs')
    require(receipt['protected_manifest_pins']['dev'] == bundle['evaluation_data']['image_manifest_sha256'],
            'Expansion used a different protected dev')
    require(receipt['total_images'] == bundle['train_images']
            and receipt['total_instances'] == bundle['train_instances'], 'Accepted train population differs')
    old, added = receipt['old_records_exact'], receipt['added_loader']
    require(old['images'] > 0 and added['images'] > 0
            and old['images']+added['images'] == receipt['total_images']
            and old['annotations']+added['instances']+added['crowd_skipped'] == receipt['total_instances'],
            'Incomplete old/new membership evidence')
    require(added['training_id_mapping'] == {str(i): i-1 for i in range(1, 9)}, 'Wrong class mapping')
    overlaps = receipt['added_cross_partition_overlap_counts']
    require(set(overlaps) == {'parent', 'dev', 'test'}, 'Missing protected partition')
    for counts in overlaps.values():
        zero_counts(counts, ('image_id', 'source_identity', 'group_id', 'sha256'))
    near = receipt['reused_audits']['near_duplicates']
    require(near['flags'] == [] and near['threshold'] == 6
            and near['comparisons'] == 'all four original/mirror orientations; new vs frozen and new vs new',
            'Missing conservative near-duplicate evidence')


def reuse_input_receipts(root, config, training_root, bundle_path):
    bundle, records, paths = read_evidence_bundle(bundle_path)
    validate_chain(bundle, records)
    require(config['purpose'] == 'dev_checkpoint_selection', 'Receipt bundle is for dev only')
    root, training_root = Path(root), Path(training_root)
    identity = lambda p: dict(image_manifest_sha256=sha256(p/'image_manifest.json'),
                              annotation_sha256=sha256(p/'annotations_coco.json'))
    require(identity(root) == bundle['evaluation_data'] == {k: config[k] for k in bundle['evaluation_data']},
            'Live dev identity changed')
    require(identity(training_root) == bundle['training_data'], 'Live train identity changed')
    if bundle['mode'] == 'unchanged_train':
        manifest, coco, check = reuse_eval_package(root, config, training_root, paths['dev_report'])
    else:
        receipt = records['acceptance']
        require(sha256(training_root/'SHA256SUMS.json') == receipt['package_pins']['SHA256SUMS.json'],
                'Accepted release catalog changed')
        manifest, coco = read_json(root/'image_manifest.json'), read_json(root/'annotations_coco.json')
        check = dict(overlap=dict(image_id=0, sha256=0, group_id=0, source_identity=0),
                     training_data=bundle['training_data'], image_and_mask_reaudit=False,
                     formal_three_party_freeze=receipt['formal_three_party_freeze'])
    require(manifest['schema_version'] == 3 and manifest['illustrative_only'] is False
            and all(im['split'] == 'dev' for im in manifest['images'])
            and len(manifest['images']) == len(coco['images']) == bundle['dev_images']
            and len(coco['annotations']) == bundle['dev_instances'], 'Wrong dev population')
    check.update(image_count=len(coco['images']), annotation_count=len(coco['annotations']),
                 validation_bundle_sha256=sha256(bundle_path), validation_mode=bundle['mode'],
                 reused_evidence={k: {'path': str(paths[k]), 'sha256': ref['sha256']}
                                  for k, ref in bundle['references'].items()},
                 image_and_mask_reaudit=False, test_files_read=False)
    return manifest, coco, check
