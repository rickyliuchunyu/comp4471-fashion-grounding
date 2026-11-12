"""COCO mask AP adapter and offline dev checkpoint selection (no language metrics)."""
import contextlib
import copy
import io
import math
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_api
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from ..data.schema import (read_json, sha256, safe_path, decode_mask, encode_mask,
                           require, unique_index)

PROTOCOL = 'coco-segm-iou50-95-maxDets100-upstream-instance-v1'


def read_eval_config(path):
    config = read_json(path)
    require(config['purpose'] in ('pilot_ap_integration', 'dev_checkpoint_selection'), 'Unknown AP purpose')
    for key in ('image_manifest_sha256', 'annotation_sha256'):
        value = config[key]
        require(isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value),
                'Pin the evaluation input: ' + key)
    for key in ('short_edge', 'max_size', 'budget_seconds'):
        require(type(config[key]) is int and config[key] > 0, 'Invalid ' + key)
    require(config['max_size'] >= config['short_edge'], 'Invalid resize bounds')
    require(config['max_detections'] == 100 and config['amp'] is False, 'AP requires maxDets=100 and FP32')
    require(config['decoder'] == 'upstream_instance', 'Language candidate decoder is not AP inference')
    return config


def check_partition(manifest, training_manifest, diagnostic):
    """Check known ID/hash/source/group overlap; this is not a near-duplicate audit."""
    for m in (manifest, training_manifest):
        require(m['schema_version'] == 3 and m['illustrative_only'] is False, 'Require real schema-3 images')
        require(bool(m['images']), 'Empty image manifest')
        unique_index(m['images'], 'image_id')
        for im in m['images']:
            require(im['source_dataset'] == 'fashionpedia', 'Fashionpedia only')
            require(type(im['image_id']) is int and im['image_id'] > 0, 'Invalid image ID')
            require(isinstance(im['group_id'], str) and bool(im['group_id']), 'Missing duplicate group')
    require(all(im['split'] == 'train' for im in training_manifest['images']), 'Training manifest must be train')
    expected = 'train' if diagnostic else 'dev'
    require(all(im['split'] == expected for im in manifest['images']), 'Expected ' + expected + ' split; test is forbidden')
    overlap = {}
    for key in ('image_id', 'sha256', 'group_id', 'source_image_id'):
        left = {im[key] for im in manifest['images']}
        right = {im[key] for im in training_manifest['images']}
        overlap[key] = len(left & right)
    require(diagnostic or not any(overlap.values()), 'Train/dev overlap: ' + str(overlap))
    return overlap


def load_eval_package(root, config, training_root):
    root, training_root = Path(root), Path(training_root)
    manifest_path, annotation_path = root/'image_manifest.json', root/'annotations_coco.json'
    require(sha256(manifest_path) == config['image_manifest_sha256'], 'Evaluation manifest hash mismatch')
    require(sha256(annotation_path) == config['annotation_sha256'], 'Evaluation annotation hash mismatch')
    manifest, coco = read_json(manifest_path), read_json(annotation_path)
    diagnostic = config['purpose'] == 'pilot_ap_integration'
    overlap = check_partition(manifest, read_json(training_root/'image_manifest.json'), diagnostic)
    images = unique_index(coco['images'], 'id')
    require(set(images) == {im['image_id'] for im in manifest['images']}, 'COCO/manifest image set mismatch')
    require(sorted(c['id'] for c in coco['categories']) == list(range(1, 9)), 'Expected eight categories')
    unique_index(coco['annotations'], 'id')
    for im in manifest['images']:
        require(all(im[k] == images[im['image_id']][k] for k in ('width', 'height', 'file_name')), 'Image metadata mismatch')
        path = safe_path(root, im['file_name'])
        require(sha256(path) == im['sha256'], 'Image hash mismatch')
        with Image.open(path) as image:
            require(image.size == (im['width'], im['height']), 'Image dimensions mismatch')
            image.convert('RGB').load()
    for ann in coco['annotations']:
        require(type(ann['id']) is int and ann['id'] > 0 and ann['image_id'] in images, 'Invalid annotation reference')
        require(type(ann['category_id']) is int and ann['category_id'] in range(1, 9), 'Invalid annotation class')
        require(ann['iscrowd'] in (0, 1), 'Invalid crowd flag')
        im = images[ann['image_id']]
        mask = decode_mask(ann['segmentation'], im['height'], im['width'])
        rle = encode_mask(mask)
        require(int(mask.sum()) == ann['area'] and np.allclose(mask_api.toBbox(rle), ann['bbox']), 'GT mask geometry mismatch')
    return manifest, coco, dict(overlap=overlap, image_count=len(images), annotation_count=len(coco['annotations']),
        training_data=dict(image_manifest_sha256=sha256(training_root/'image_manifest.json'),
                           annotation_sha256=sha256(training_root/'annotations_coco.json')))


def evaluate_masks(coco, rows):
    """All images retained, including no detections and explicit inference failures.

    Rows contain upstream top-k masks, not filtered language candidates. Omit bbox
    from loadRes so COCO derives result area from masks for AP-small/medium/large.
    """
    images = unique_index(coco['images'], 'id')
    indexed = unique_index(rows, 'image_id')
    require(set(indexed) == set(images), 'Every GT image needs an evaluation row')
    results, errors = [], 0
    for iid, row in indexed.items():
        detections = row['detections']
        require(row['status'] in ('ok', 'no_detections', 'inference_error'), 'Bad inference status')
        require((row['status'] == 'ok') == bool(detections), 'Status/detection mismatch')
        require(len(detections) <= 100, 'Too many AP detections')
        if row['status'] == 'inference_error':
            require(bool(row.get('error')), 'Missing failure reason')
            errors += 1
        for detection in detections:
            require(detection['image_id'] == iid, 'Prediction image mismatch')
            require(type(detection['category_id']) is int and detection['category_id'] in range(1, 9), 'Prediction class mismatch')
            require(math.isfinite(detection['score']) and 0 <= detection['score'] <= 1, 'Invalid score')
            rle = detection['segmentation']
            require(rle['size'] == [images[iid]['height'], images[iid]['width']], 'Prediction mask dimensions')
            # Upstream can emit empty masks with zero score: retain them for AP.
            mask = mask_api.decode(rle)
            require(mask.shape == tuple(rle['size']), 'Bad prediction RLE')
            results.append({k: detection[k] for k in ('image_id', 'category_id', 'score', 'segmentation')})
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        gt = COCO()
        gt.dataset = copy.deepcopy(coco)
        gt.dataset.setdefault('info', {})
        gt.createIndex()
        if results:
            dt = gt.loadRes(results)
        else:
            # pycocotools loadRes([]) indexes element zero; build its empty equivalent.
            dt = COCO()
            dt.dataset = dict(images=copy.deepcopy(coco['images']), categories=copy.deepcopy(coco['categories']), annotations=[])
            dt.createIndex()
        evaluator = COCOeval(gt, dt, 'segm')
        evaluator.params.imgIds = sorted(images)
        evaluator.params.catIds = list(range(1, 9))
        evaluator.params.maxDets = [1, 10, 100]
        evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
    def scalar(value):
        return None if value < 0 else float(value) * 100
    metrics = {name: scalar(evaluator.stats[i]) for i, name in enumerate(('AP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large'))}
    precision = evaluator.eval['precision']
    metrics['per_category_AP'] = {}
    for k, category in enumerate(evaluator.params.catIds):
        values = precision[:, :, k, 0, -1]
        values = values[values > -1]
        metrics['per_category_AP'][str(category)] = float(values.mean()*100) if values.size else None
    return dict(protocol=PROTOCOL, metric_units='percent', metrics=metrics,
                evaluated_images=len(images), detections=len(results), inference_errors=errors,
                complete=errors == 0, coco_summary=capture.getvalue())


def choose_checkpoint(reports):
    """Pure selection rule: maximum full-dev mask AP, tie -> earliest update."""
    require(bool(reports), 'No evaluation reports')
    first, seen = reports[0], set()
    comparison = ('protocol', 'evaluation_config_sha256', 'evaluation_data', 'training_data',
                  'training_config_sha256', 'training_git_commit', 'evaluation_git_commit')
    for report in reports:
        require(report['purpose'] == 'dev_checkpoint_selection' and report['selection_eligible'] is True,
                'Only complete dev reports may select a checkpoint; pilot diagnostics are forbidden')
        require(report['complete'] is True and report['inference_errors'] == 0, 'Incomplete evaluation')
        require(report['protocol'] == PROTOCOL and report['metric_units'] == 'percent', 'Wrong AP protocol')
        require(all(report[k] == first[k] for k in comparison), 'Incomparable checkpoint evaluations')
        score, update = report['metrics']['AP'], report['completed_updates']
        require(isinstance(score, (int, float)) and math.isfinite(score) and 0 <= score <= 100, 'AP is undefined/invalid')
        require(type(update) is int and update > 0 and update not in seen, 'Duplicate/invalid checkpoint update')
        seen.add(update)
    best = min(reports, key=lambda r: (-r['metrics']['AP'], r['completed_updates']))
    return dict(rule='max_dev_mask_AP_then_earliest_update', checkpoint_sha256=best['checkpoint_sha256'],
                checkpoint_path=best['checkpoint_path'], completed_updates=best['completed_updates'],
                dev_mask_AP=best['metrics']['AP'], compared_updates=sorted(seen))
