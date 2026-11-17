"""CPU, train-only augmentation visibility. No RGB audit or model execution."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import random
from pathlib import Path
import time

import numpy as np
from pycocotools import mask as mask_api

from ..data.schema import sha256
from .data import training_transforms


def select_images(images, annotations, old_ids, seed, uniform=128, targeted=256):
    """Keep a separately reported uniform sample; enrich weak/small train strata."""
    rng = np.random.RandomState(seed)
    ids = sorted(im['image_id'] for im in images)
    uniform_ids = set(rng.choice(ids, min(uniform, len(ids)), replace=False).tolist())
    pools = defaultdict(set)
    for ann in annotations:
        if not ann.get('iscrowd', 0) and ann['area'] <= 1024:
            pools[(ann['image_id'] in old_ids, ann.get('source_category_name', str(ann['category_id'])))].add(ann['image_id'])
    queues = [list(rng.permutation(sorted(pools[k]))) for k in sorted(pools)]
    extra = set()
    while len(extra) < targeted and any(queues):
        for queue in queues:
            while queue:
                image_id = int(queue.pop())
                if image_id not in uniform_ids | extra:
                    extra.add(image_id)
                    break
            if len(extra) == targeted:
                break
    return {i: ('uniform' if i in uniform_ids else 'targeted') for i in sorted(uniform_ids | extra)}


def mask_visibility(mask, transforms):
    """Crop retention uses resized foreground, not incomparable source pixel area."""
    current, retention, resized_area, vanished = mask, 1., int(mask.sum()), False
    for transform in transforms.transforms:
        before = int(current.astype(bool).sum())
        current = transform.apply_segmentation(current)
        after = int(current.astype(bool).sum())
        if type(transform).__name__ == 'CropTransform':
            retention *= after/before if before else 0.
        if type(transform).__name__ == 'ResizeTransform':
            resized_area = after
            vanished |= before > 0 and after == 0
    final = current.astype(bool)
    area = int(final.sum())
    ys, xs = np.nonzero(final)
    retention = retention if area else 0.
    return dict(resized_area=resized_area, augmented_area=area,
                retained_fraction=retention,
                disappeared=area == 0, vanished_during_resize=vanished,
                retained_below_half=retention < .5,
                min_bbox_side=int(min(xs.max()-xs.min()+1, ys.max()-ys.min()+1)) if area else 0)


def aggregate(rows):
    if not rows:
        return {'observations': 0}
    result = dict(observations=len(rows), unique_instances=len({r['annotation_id'] for r in rows}),
                  unique_images=len({r['image_id'] for r in rows}))
    for field in ('disappeared', 'vanished_during_resize', 'retained_below_half'):
        result[field+'_count'] = sum(r[field] for r in rows)
        result[field+'_rate'] = result[field+'_count'] / len(rows)
    for field in ('augmented_area', 'retained_fraction', 'min_bbox_side'):
        result[field+'_quantiles'] = dict(zip(('p10', 'p50', 'p90'),
                                             map(float, np.quantile([r[field] for r in rows], [.1, .5, .9]))))
    result['area_below_64_rate'] = sum(r['augmented_area'] < 64 for r in rows) / len(rows)
    return result


def diagnose(package, config, parent_manifest, output, seed=20261117, views=4, limit_seconds=900):
    import torch
    torch.set_num_threads(2)
    started = time.monotonic()
    package, output = Path(package), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    identity = dict(image_manifest_sha256=sha256(package/'image_manifest.json'),
                    annotation_sha256=sha256(package/'annotations_coco.json'))
    if identity != {k: config[k] for k in identity}:
        raise ValueError('Diagnostic data differs from pinned training input')
    manifest = json.loads((package/'image_manifest.json').read_text())
    coco = json.loads((package/'annotations_coco.json').read_text())
    if not all(im['split'] == 'train' and im['source_dataset'] == 'fashionpedia' for im in manifest['images']):
        raise ValueError('Only Fashionpedia train may inform augmentation')
    old_ids = {im['image_id'] for im in json.loads(Path(parent_manifest).read_text())['images']}
    selected = select_images(manifest['images'], coco['annotations'], old_ids, seed)
    images = {im['image_id']: im for im in manifest['images']}
    anns = defaultdict(list)
    for ann in coco['annotations']:
        if ann['image_id'] in selected and not ann.get('iscrowd', 0):
            anns[ann['image_id']].append(ann)
    rows = []
    for image_id, cohort in selected.items():
        im = images[image_id]
        masks = [(ann, mask_api.decode(ann['segmentation'])) for ann in anns[image_id]]
        # LSJ samples solely from shape/RNG; zeros avoid repeating image decode checks.
        canvas = np.zeros((im['height'], im['width'], 3), dtype=np.uint8)
        for view in range(views):
            if time.monotonic()-started > limit_seconds:
                raise TimeoutError('Visibility diagnosis exceeded fixed CPU time budget')
            draw_seed = int.from_bytes(hashlib.sha256(f'{seed}:{image_id}:{view}'.encode()).digest()[:4], 'little')
            np.random.seed(draw_seed)
            random.seed(draw_seed)
            _, transforms = training_transforms(canvas, config, anns[image_id])
            for ann, mask in masks:
                rows.append(dict(image_id=image_id, annotation_id=ann['id'], cohort=cohort,
                    membership='old5k' if image_id in old_ids else 'added5k', view=view,
                    category_id=ann['category_id'], source_category=ann.get('source_category_name', 'unknown'),
                    source_area=int(ann['area']), view_kind=transforms.view_kind,
                    selected_target=ann['id'] == transforms.target_annotation_id,
                    **mask_visibility(mask, transforms)))
    groups = defaultdict(list)
    for row in rows:
        size = 'small' if row['source_area'] <= 1024 else 'non_small'
        for key in (f"{row['cohort']}/{size}", f"all/{size}",
                    f"{row['membership']}/{size}", f"fine/{row['source_category']}/{size}"):
            groups[key].append(row)
        if row['source_area'] <= 256:
            groups[f"{row['cohort']}/extreme"].append(row)
        if row['selected_target']:
            groups['selected_targets'].append(row)
        if row['view_kind'] == 'target_context':
            groups[f"context_views/{size}"].append(row)
    report = dict(status='completed', checked_at=datetime.now(timezone.utc).isoformat(),
        input=identity, config=config, seed=seed, views=views, selected_images=len(selected),
        parent_manifest_sha256=sha256(parent_manifest), image_ids_by_cohort={
            c:[i for i in selected if selected[i] == c] for c in ('uniform', 'targeted')},
        groups={k:aggregate(v) for k,v in sorted(groups.items())}, elapsed_seconds=time.monotonic()-started,
        scope='train-only geometry; uniform and enriched cohorts separate; repeated views are not independent images',
        model_loads=0, training_updates=0, inference_images=0, rgb_files_read=0, dev_or_test_read=False,
        limitation='Does not measure AP, semantic annotation correctness, or actual loss-point coverage')
    report['target_context_views'] = len({(r['image_id'],r['view']) for r in rows if r['view_kind']=='target_context'})
    (output/'observations.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    report['observations_sha256'] = sha256(output/'observations.jsonl')
    (output/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('package', 'config', 'parent-manifest', 'output'):
        parser.add_argument('--'+key, required=True)
    args = parser.parse_args()
    result = diagnose(args.package, json.loads(Path(args.config).read_text()), args.parent_manifest, args.output)
    print(json.dumps({k:result[k] for k in ('status', 'selected_images', 'elapsed_seconds', 'groups')}))


if __name__ == '__main__':
    main()
