"""Pure metadata contracts for full-data training; no image/model imports."""
import hashlib
import json
import math
from pathlib import Path

from .full_data_preparation import epoch_schedule, read_json, require

PROTOCOL = 'coco-segm-iou50-95-maxDets100-upstream-instance-v1'
IDENTITY_KEYS = ('image_manifest_sha256', 'annotation_sha256')
OVERLAP_KEYS = {'image_id', 'source_identity', 'group_id', 'sha256'}
FREEZE_CHECKS = {'source_inventory', 'mapping', 'source_identity', 'quality_disposition',
                 'split_isolation', 'protocol_amendment', 'population_statistics',
                 'formal_freeze', 'consumer_acceptance', 'dev_version'}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def digest_json(value):
    # Same encoding as the existing training_state.config_sha256.
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def is_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def referenced(base, ref):
    require(isinstance(ref, dict) and isinstance(ref.get('path'), str)
            and bool(ref['path']) and is_sha(ref.get('sha256')), 'Unresolved evidence reference')
    path = (Path(base) / ref['path']).resolve()
    require(path.is_file() and sha256(path) == ref['sha256'], 'Evidence changed: ' + ref['path'])
    return path, read_json(path)


def package_metadata(path, expected, split, categories):
    """Verify immutable JSON identities, never re-audit pixels or decode masks."""
    path = Path(path).resolve()
    manifest_path, coco_path = path/'image_manifest.json', path/'annotations_coco.json'
    require({k: sha256(p) for k, p in zip(IDENTITY_KEYS, (manifest_path, coco_path))} == expected,
            'Live package identity differs: ' + split)
    manifest, coco = read_json(manifest_path), read_json(coco_path)
    require(manifest.get('schema_version') == 3 and manifest.get('illustrative_only') is False,
            'Expected real schema-3 package')
    rows = manifest['images']
    ids = [im['image_id'] for im in rows]
    require(rows and all(type(i) is int and i > 0 for i in ids)
            and len(ids) == len(set(ids)), 'Invalid manifest population')
    require(all(im['split'] == split and im['source_dataset'] == 'fashionpedia' for im in rows),
            'Wrong source or partition')
    require(all(is_sha(im.get('sha256')) and isinstance(im.get('group_id'), str) and im['group_id']
                and isinstance(im.get('source_split'), str) and im['source_split']
                and isinstance(im.get('source_image_id'), str) and im['source_image_id'] for im in rows),
            'Incomplete image provenance')
    images = {im['id']: im for im in coco['images']}
    require(len(images) == len(coco['images']) and set(images) == set(ids), 'COCO population differs')
    require(len(coco['categories']) == 8 and
            {c['id']: c['name'] for c in coco['categories']} == categories, 'Wrong class mapping')
    for im in rows:
        require(all(images[im['image_id']][k] == im[k] for k in ('width', 'height', 'file_name')),
                'COCO image metadata differs')
        target = (path / im['file_name']).resolve()
        require(target.is_relative_to(path), 'Image path escapes package')
    ann_ids = [a['id'] for a in coco['annotations']]
    require(all(type(i) is int and i > 0 for i in ann_ids) and len(ann_ids) == len(set(ann_ids)),
            'Invalid annotation IDs')
    require(all(a['image_id'] in images and type(a['category_id']) is int
                and a['category_id'] in categories and a.get('iscrowd', 0) in (0, 1)
                for a in coco['annotations']), 'Invalid annotation identity')
    for ann in coco['annotations']:
        segmentation, im = ann.get('segmentation'), images[ann['image_id']]
        require(isinstance(segmentation, dict) and isinstance(segmentation.get('counts'), str)
                and segmentation.get('size') == [im['height'], im['width']],
                'Publish accepted original-size compressed RLE for metadata-only training reuse')
    return manifest, coco


def load_binding(path, mapping_sha, categories, expected_dev):
    path = Path(path).resolve()
    binding = read_json(path)
    require(binding.get('schema_version') == 1 and binding.get('purpose') == 'full100_data_binding',
            'Wrong data binding')
    require(all(isinstance(binding.get(k), str) and binding[k] for k in ('train_package', 'dev_package')),
            'Full training package has not been released')
    require(set(binding['references']) == {'freeze', 'acceptance', 'dev_report'}, 'Incomplete data evidence')
    evidence = {name: referenced(path.parent, ref) for name, ref in binding['references'].items()}
    freeze_path, freeze = evidence['freeze']
    receipt = evidence['acceptance'][1]
    dev_receipt = evidence['dev_report'][1]
    require(freeze.get('formal_freeze') is True and freeze.get('status') == 'frozen'
            and freeze.get('dataset') == 'fashionpedia' and freeze.get('source_split') == 'train',
            'Full-data formal freeze is incomplete')
    require(freeze['mapping']['sha256'] == mapping_sha, 'Frozen mapping differs')
    referenced(freeze_path.parent, freeze['mapping'])
    accounting = freeze['source_accounting']
    require(accounting['coverage'] == 'all_eligible_official_train'
            and all(type(accounting[k]) is int and accounting[k] >= 0
                    for k in ('source_images', 'retained_images', 'excluded_images'))
            and accounting['source_images'] == accounting['retained_images'] + accounting['excluded_images']
            and accounting['retained_images'] == freeze['actual_train_images'] > 0,
            'Incomplete full-source accounting')
    checks = {c['id']: c for c in freeze['checks']}
    require(len(checks) == len(freeze['checks']) and set(checks) == FREEZE_CHECKS,
            'Missing freeze checks')
    for check in checks.values():
        require(check['status'] == 'passed', 'Unresolved freeze condition: ' + check['id'])
        referenced(freeze_path.parent, check['evidence'])
    require(checks['consumer_acceptance']['evidence']['sha256'] ==
            binding['references']['acceptance']['sha256'], 'Freeze binds a different consumer receipt')
    require(set(freeze['acknowledgements']) == {'wkq', 'lcy', 'wyz'}, 'Missing member freeze records')
    for member, acknowledgement in freeze['acknowledgements'].items():
        _, record = referenced(freeze_path.parent, acknowledgement)
        require(record.get('member') == member and record.get('decision') == 'approved'
                and record.get('image_manifest_sha256') == freeze['image_manifest_sha256']
                and record.get('annotation_sha256') == freeze['annotation_sha256']
                and isinstance(record.get('reviewed_at'), str) and record['reviewed_at']
                and isinstance(record.get('reviewer_identity'), str) and record['reviewer_identity'],
                'Invalid member acknowledgement')
    train_identity = {k: freeze[k] for k in IDENTITY_KEYS}
    require(all(is_sha(v) for v in train_identity.values()), 'Unpinned full-data identity')
    require(receipt.get('status') == 'passed' and receipt.get('scope') == 'full_fashionpedia_segmentation_input'
            and receipt.get('training_data') == train_identity
            and receipt.get('evaluation_data') == expected_dev
            and receipt.get('mapping_sha256') == mapping_sha, 'Incompatible full-data acceptance')
    require(receipt['train_images'] == freeze['actual_train_images']
            and receipt['train_instances'] == freeze['actual_train_instances'], 'Acceptance population differs')
    commit = receipt.get('source_commit')
    require(isinstance(commit, str) and len(commit) == 40
            and all(c in '0123456789abcdef' for c in commit), 'Missing acceptance source commit')
    require(freeze['protected']['existing_dev_manifest_sha256'] == expected_dev['image_manifest_sha256']
            and freeze['protected']['existing_dev_annotation_sha256'] == expected_dev['annotation_sha256'],
            'Freeze binds a different dev version')
    require(receipt['image_and_mask_acceptance_complete'] is True
            and receipt['near_duplicate_review_complete'] is True, 'Incomplete incremental acceptance')
    require(receipt['training_id_mapping'] == {str(i): i-1 for i in range(1, 9)}, 'Consumer training classes differ')
    require(receipt['protected_test_manifest_sha256'] == freeze['protected']['test_manifest_sha256'],
            'Protected test identity differs')
    for partition in ('dev', 'test'):
        counts = receipt['overlap'][partition]
        require(set(counts) == OVERLAP_KEYS and all(type(v) is int and v == 0 for v in counts.values()),
                'Missing or nonzero partition evidence')
    require(dev_receipt['purpose'] == 'dev_checkpoint_selection' and dev_receipt['protocol'] == PROTOCOL
            and dev_receipt['complete'] is True and dev_receipt['selection_eligible'] is True
            and dev_receipt['inference_errors'] == 0 and dev_receipt['evaluation_data'] == expected_dev,
            'Prior dev validation is incompatible')
    train_root, dev_root = Path(binding['train_package']).resolve(), Path(binding['dev_package']).resolve()
    train_manifest, train_coco = package_metadata(train_root, train_identity, 'train', categories)
    dev_manifest, dev_coco = package_metadata(dev_root, expected_dev, 'dev', categories)
    def values(manifest, kind):
        if kind == 'source_identity':
            return {(im['source_dataset'], im['source_split'], im['source_image_id']) for im in manifest['images']}
        return {im[kind] for im in manifest['images']}
    require(all(not (values(train_manifest, key) & values(dev_manifest, key)) for key in OVERLAP_KEYS),
            'Live train/dev metadata overlap')
    require(len(train_manifest['images']) == receipt['train_images']
            and len(train_coco['annotations']) == receipt['train_instances'], 'Live train count differs')
    require(len(dev_manifest['images']) == receipt['dev_images'] == dev_receipt['evaluated_images']
            and len(dev_coco['annotations']) == receipt['dev_instances'], 'Live dev count differs')
    require(sha256(train_root/'SHA256SUMS.json') == receipt['catalog_sha256'], 'Release catalog changed')
    return dict(binding_path=str(path), binding_sha256=sha256(path), training_data=train_identity,
                evaluation_data=expected_dev, train_package=str(train_root), dev_package=str(dev_root),
                train_images=receipt['train_images'], train_instances=receipt['train_instances'],
                dev_images=receipt['dev_images'], dev_instances=receipt['dev_instances'],
                evidence_hashes={k: v['sha256'] for k, v in binding['references'].items()},
                overlap=receipt['overlap']['dev'], image_and_mask_reaudit=False), (
                    train_manifest, train_coco, dev_manifest, dev_coco)


def resolve(root, plan_path, binding_path, source_commit):
    root, plan_path = Path(root).resolve(), Path(plan_path).resolve()
    plan = read_json(plan_path)
    require(plan['purpose'] == 'full100_execution_plan' and plan['execute_by_default'] is False
            and plan['total_wall_seconds'] is None and plan['automatic_retry'] is False
            and plan['automatic_epoch_extension'] is False, 'Invalid full100 execution policy')
    objects = {k: referenced(root, plan[k])[1] for k in ('recipe', 'evaluation', 'mapping', 'categories', 'upstream')}
    for ref in plan['legacy_execution_files']:
        referenced(root, ref) if ref['path'].endswith('.json') else require(
            sha256(root/ref['path']) == ref['sha256'], 'Legacy execution source changed')
    recipe, evaluation = objects['recipe'], objects['evaluation']
    require(recipe['model'] == 'Mask2Former-R50' and recipe['epochs'] == 100
            and recipe['checkpoint_every_epochs'] == 1 and recipe['evaluation_every_epochs'] == 2,
            'Keep R50, 100 epochs and 2-epoch dev cadence')
    expected_dev = {k: evaluation['evaluation_config_template'][k] for k in IDENTITY_KEYS}
    result = dict(schema_version=1, purpose='full100_resolved_contract', source_commit=source_commit,
                  plan_path=str(plan_path), plan_sha256=sha256(plan_path),
                  runtime=plan, ready_for_training=False, requires_explicit_execute=True,
                  remaining=['full_dataset_freeze_and_acceptance_binding', 'launch_resource_check_and_authorization'])
    if binding_path is None:
        return result
    categories = {c['id']: c['name'] for c in objects['categories']['categories']}
    inputs, _ = load_binding(binding_path, plan['mapping']['sha256'], categories, expected_dev)
    require(inputs['dev_images'] == evaluation['dev_images']
            and inputs['dev_instances'] == evaluation['dev_instances'], 'Pinned dev population changed')
    schedule = epoch_schedule(inputs['train_images'], recipe)
    train = dict(recipe['train_config_template'], **inputs['training_data'], purpose='s0_full100_training',
                 input_binding_sha256=inputs['binding_sha256'])
    train['run_scope'] = 'full_fashionpedia_r50_100ep'
    for key in ('max_updates', 'lr_milestones', 'checkpoint_interval', 'checkpoint_updates'):
        train[key] = schedule[key]
    require(train['microbatch_images'] * train['accumulation_steps'] == train['effective_batch_images'] == 16,
            'Wrong full100 batch')
    dev = dict(evaluation['evaluation_config_template'], input_binding_sha256=inputs['binding_sha256'])
    dev['run_scope'] = 'full_fashionpedia_100ep_complete_dev'
    result.update(inputs=inputs, schedule=schedule, train_config=train, eval_config=dev,
                  train_config_sha256=digest_json(train), eval_config_sha256=digest_json(dev),
                  upstream=objects['upstream'], ready_for_training=True,
                  remaining=['launch_resource_check_and_authorization'])
    return result


def cache_identity(contract, checkpoint):
    return dict(checkpoint_sha256=checkpoint['sha256'], completed_updates=checkpoint['update'],
                training_data=contract['inputs']['training_data'],
                evaluation_data=contract['inputs']['evaluation_data'],
                training_config_sha256=contract['train_config_sha256'],
                evaluation_config_sha256=contract['eval_config_sha256'],
                training_git_commit=contract['source_commit'], evaluation_git_commit=contract['source_commit'],
                input_binding_sha256=contract['inputs']['binding_sha256'])


def validate_report(report, contract, checkpoint):
    require(all(report.get(k) == v for k, v in cache_identity(contract, checkpoint).items()),
            'Evaluation cache identity changed')
    require(report.get('purpose') == 'dev_checkpoint_selection' and report.get('protocol') == PROTOCOL
            and report.get('metric_units') == 'percent' and report.get('complete') is True
            and report.get('selection_eligible') is True and report.get('inference_errors') == 0
            and report.get('evaluated_images') == contract['inputs']['dev_images'], 'Incomplete dev result')
    ap = report['metrics']['AP']
    require(type(ap) in (int, float) and math.isfinite(ap) and 0 <= ap <= 100, 'Invalid AP')
    return report


def choose(reports):
    require(bool(reports), 'No completed full dev reports')
    winner = min(reports, key=lambda r: (-r['metrics']['AP'], r['completed_updates']))
    return dict(rule='max_dev_mask_AP_then_earliest_update',
                checkpoint_path=winner['checkpoint_path'], checkpoint_sha256=winner['checkpoint_sha256'],
                completed_updates=winner['completed_updates'], dev_mask_AP=winner['metrics']['AP'],
                compared_updates=sorted(r['completed_updates'] for r in reports))
