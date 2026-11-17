"""CPU-only preparation of the full-data recipe; never imports or launches a trainer.

Metadata binding is not data acceptance or formal freeze. All outputs deliberately
remain non-executable until the separately documented launch requirements are met.
"""
import copy
import hashlib
import json
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def positive_integer(value, name):
    require(type(value) is int and value > 0, f'{name} must be a positive integer')


def checked_reference(root, reference):
    root = Path(root).resolve()
    path = (root / reference['path']).resolve()
    require(path.is_relative_to(root), 'Reference escapes repository')
    require(file_hash(path) == reference['sha256'], f'Changed pinned file: {reference["path"]}')
    return path


def epoch_schedule(images, recipe):
    """Round cumulative image exposure once, not ceil(N/B) separately each epoch."""
    positive_integer(images, 'images')
    epochs = recipe['epochs']
    batch = recipe['train_config_template']['effective_batch_images']
    positive_integer(epochs, 'epochs')
    positive_integer(batch, 'effective_batch_images')
    checkpoint_period = recipe['checkpoint_every_epochs']
    evaluation_period = recipe['evaluation_every_epochs']
    positive_integer(checkpoint_period, 'checkpoint_every_epochs')
    positive_integer(evaluation_period, 'evaluation_every_epochs')
    milestones = recipe['milestone_epochs']
    require(isinstance(milestones, list) and milestones == sorted(set(milestones)),
            'Invalid milestone order')
    require(all(type(e) is int and 0 < e < epochs for e in milestones),
            'Milestone must be an interior epoch')
    require(images >= batch, 'Dataset smaller than effective batch is outside full-data preparation')

    def update(epoch):
        return (epoch * images + batch - 1) // batch

    checkpoints = sorted(set(range(checkpoint_period, epochs + 1, checkpoint_period)) | {epochs})
    evaluations = sorted(set(range(evaluation_period, epochs + 1, evaluation_period)) | {epochs})
    require(set(evaluations) <= set(checkpoints), 'Every evaluation requires a saved checkpoint')
    milestone_updates = [update(e) for e in milestones]
    require(milestone_updates and recipe['train_config_template']['warmup_updates'] < milestone_updates[0],
            'Warmup must finish before the first learning-rate decay')
    return {
        'images': images, 'epochs': epochs, 'effective_batch_images': batch,
        'max_updates': update(epochs), 'lr_milestones': milestone_updates,
        'checkpoint_interval': update(checkpoint_period),
        'checkpoint_updates': [update(e) for e in checkpoints],
        'evaluation_nodes': [{'epoch': e, 'update': update(e)} for e in evaluations],
        'successful_image_exposures': update(epochs) * batch,
        'rounding_extra_images': update(epochs) * batch - epochs * images,
        'checkpoint_interval_is_fallback_only': True,
        'lr_decay_applies_after_completed_milestone': True,
    }


def metadata_identity(manifest_path, annotation_path, categories):
    """Read JSON only. No images, mask decoding, duplicate search, or test data."""
    manifest = read_json(manifest_path)
    coco = read_json(annotation_path)
    require(manifest.get('schema_version') == 3 and manifest.get('illustrative_only') is False,
            'Expected real schema-3 manifest')
    rows = manifest.get('images')
    require(isinstance(rows, list) and rows, 'Empty manifest')
    ids = [row['image_id'] for row in rows]
    require(all(type(i) is int and i > 0 for i in ids) and len(ids) == len(set(ids)),
            'Duplicate or invalid manifest IDs')
    require(all(row.get('split') == 'train' and row.get('source_dataset') == 'fashionpedia'
                for row in rows), 'Only Fashionpedia train metadata can determine the schedule')
    coco_ids = [row['id'] for row in coco['images']]
    require(len(coco_ids) == len(set(coco_ids)) and set(coco_ids) == set(ids),
            'COCO/manifest population mismatch')
    expected = {int(row['id']): row['name'] for row in categories}
    actual = {row['id']: row['name'] for row in coco['categories']}
    require(len(coco['categories']) == 8 and actual == expected,
            'Project category identities differ')
    anns = coco['annotations']
    ann_ids = [row['id'] for row in anns]
    require(all(type(i) is int and i > 0 for i in ann_ids) and len(ann_ids) == len(set(ann_ids)),
            'Duplicate or invalid annotation IDs')
    population = set(ids)
    require(all(row['image_id'] in population and type(row['category_id']) is int
                and row['category_id'] in expected for row in anns), 'Unknown annotation identity')
    require(all(type(row.get('iscrowd', 0)) is int and row.get('iscrowd', 0) in (0, 1)
                for row in anns), 'Invalid crowd flag')
    return {
        'basis': 'metadata_only_not_acceptance_or_freeze', 'images': len(rows),
        'annotations_including_crowd': len(anns),
        'noncrowd_instances': sum(not row.get('iscrowd', 0) for row in anns),
        'image_manifest_sha256': file_hash(manifest_path),
        'annotation_sha256': file_hash(annotation_path),
    }


def prepare(root, plan_path, *, manifest=None, annotations=None, illustrative_images=None):
    root = Path(root)
    plan = read_json(plan_path)
    require(plan.get('purpose') == 'full100_preparation_only'
            and plan.get('execution_enabled') is False
            and plan.get('training_authorized') is False, 'Not a preparation-only plan')
    require((manifest is None) == (annotations is None), 'Supply both manifest and annotations')
    require(illustrative_images is None or manifest is None, 'Cannot mix illustrative and real metadata')
    paths = {name: checked_reference(root, ref) for name, ref in plan['references'].items()}
    for ref in plan['protected_existing_files']:
        checked_reference(root, ref)
    recipe = read_json(paths['recipe'])
    evaluation = read_json(paths['evaluation'])
    recovery = read_json(paths['recovery'])
    freeze = read_json(paths['freeze_checklist'])
    train = copy.deepcopy(recipe['train_config_template'])
    require(recipe['model'] == 'Mask2Former-R50' and recipe['epochs'] == 100,
            'This preparation is fixed to existing R50 and 100 epochs')
    require(train['microbatch_images'] * train['accumulation_steps'] == train['effective_batch_images'],
            'Effective batch mismatch')
    require(train['budget_seconds'] is None, 'Do not inherit the old wall-time limit')
    require(train['max_updates'] is None and train['image_manifest_sha256'] is None
            and train['annotation_sha256'] is None, 'Recipe must not contain invented full-data identity')
    require(evaluation['evaluation_every_epochs'] == recipe['evaluation_every_epochs'],
            'Evaluation frequency mismatch')
    dev = evaluation['evaluation_config_template']
    require(dev['image_manifest_sha256'] == freeze['protected']['existing_dev_manifest_sha256']
            and dev['annotation_sha256'] == freeze['protected']['existing_dev_annotation_sha256'],
            'Dev identity mismatch')
    require(recovery['scheduler_reset'] is False and recovery['reinitialize_optimizer'] is False,
            'Recovery must preserve optimization state')
    require(freeze['formal_freeze'] is False, 'Use an independently reviewed release process for actual freeze')
    identity = {'basis': 'unresolved_full_release', 'images': None,
                'image_manifest_sha256': None, 'annotation_sha256': None}
    if illustrative_images is not None:
        positive_integer(illustrative_images, 'illustrative_images')
        identity.update(basis='illustrative_arithmetic_only', images=illustrative_images)
    elif manifest is not None:
        categories = read_json(paths['categories'])
        if isinstance(categories, dict):
            categories = categories['categories']
        identity = metadata_identity(manifest, annotations, categories)
    schedule = epoch_schedule(identity['images'], recipe) if identity['images'] is not None else None
    if schedule:
        for key in ('max_updates', 'lr_milestones', 'checkpoint_interval', 'checkpoint_updates'):
            train[key] = schedule[key]
        for key in ('image_manifest_sha256', 'annotation_sha256'):
            train[key] = identity[key]
    return {
        'schema_version': 1, 'status': 'prepared_not_executable',
        'training_authorized': False, 'ready_for_training': False,
        'plan_sha256': file_hash(plan_path), 'input_references': plan['references'],
        'protected_files_unchanged': True, 'data_identity': identity, 'schedule': schedule,
        'train_config_preparation': train, 'evaluation_plan': evaluation,
        'recovery_plan': recovery, 'freeze_status': freeze['status'],
        'remaining_requirements': plan['release_blockers'],
        'operations': {'metadata_only': True, 'image_reads': 0, 'mask_decodes': 0,
                       'model_loads': 0, 'training_updates': 0, 'inference_images': 0,
                       'server_connections': 0},
    }


def write_preparation(output, report):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name, value in [('preparation.json', report),
                        ('train_config.preparation.json', report['train_config_preparation'])]:
        with (output / name).open('x', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
