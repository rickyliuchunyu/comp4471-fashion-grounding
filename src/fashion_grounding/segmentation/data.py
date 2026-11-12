"""Consumer adapter for wkq's converted COCO package, not a source converter."""
import copy
import json
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_api

from .predict import safe_path, sha256


def load_training_package(root):
    """Validate the real train-only package before producing Detectron2 records."""
    from detectron2.structures import BoxMode
    root = Path(root)
    manifest = json.loads((root / 'image_manifest.json').read_text(encoding='utf-8'))
    coco = json.loads((root / 'annotations_coco.json').read_text(encoding='utf-8'))
    if manifest['schema_version'] != 3 or manifest['illustrative_only'] is not False:
        raise ValueError('This loader requires a real schema-3 training package')
    if sorted(c['id'] for c in coco['categories']) != list(range(1, 9)):
        raise ValueError('Expected exactly project categories 1..8')
    images = {im['id']: im for im in coco['images']}
    mids = [im['image_id'] for im in manifest['images']]
    if len(images) != len(coco['images']) or len(set(mids)) != len(mids) or set(images) != set(mids):
        raise ValueError('Duplicate/mismatched image IDs')
    annotations, annotation_ids = {}, set()
    for ann in coco['annotations']:
        if ann['id'] in annotation_ids or ann['image_id'] not in images or ann['id'] <= 0:
            raise ValueError('Duplicate/invalid annotation ID or unknown image')
        annotation_ids.add(ann['id'])
        annotations.setdefault(ann['image_id'], []).append(ann)
    records, counts = [], {str(i): 0 for i in range(1, 9)}
    crowd_skipped = 0
    for im in manifest['images']:
        if im['split'] != 'train' or im['source_dataset'] != 'fashionpedia':
            raise ValueError('Smoke training only accepts Fashionpedia train images')
        ci = images[im['image_id']]
        if any(ci[k] != im[k] for k in ('file_name', 'width', 'height')):
            raise ValueError('COCO/manifest mismatch')
        path = safe_path(root, im['file_name'])
        if sha256(path) != im['sha256']:
            raise ValueError('Image hash mismatch')
        with Image.open(path) as image:
            if image.size != (im['width'], im['height']):
                raise ValueError('Image size mismatch')
            image.convert('RGB').load()
        anns = []
        for ann in sorted(annotations.get(im['image_id'], []), key=lambda a: a['id']):
            category = ann['category_id']
            if type(category) is not int or not 1 <= category <= 8:
                raise ValueError('Expected project category 1..8')
            segmentation = ann['segmentation']
            if isinstance(segmentation, list):
                rle = mask_api.merge(mask_api.frPyObjects(segmentation, im['height'], im['width']))
            elif isinstance(segmentation['counts'], list):
                rle = mask_api.frPyObjects(segmentation, im['height'], im['width'])
            else:
                rle = segmentation
            mask = mask_api.decode(rle)
            if mask.shape != (im['height'], im['width']) or not mask.any():
                raise ValueError('Invalid mask')
            if not np.allclose(mask_api.toBbox(rle), ann['bbox']) or int(mask.sum()) != ann['area']:
                raise ValueError('Mask/COCO geometry mismatch')
            if ann.get('iscrowd', 0):
                crowd_skipped += 1
                continue
            anns.append(dict(bbox=ann['bbox'], bbox_mode=BoxMode.XYWH_ABS,
                             segmentation=rle, category_id=category-1, iscrowd=0))
            counts[str(category)] += 1
        records.append(dict(file_name=str(path), image_id=im['image_id'], height=im['height'],
                            width=im['width'], annotations=anns))
    report = dict(images=len(records), instances=sum(counts.values()), project_category_counts=counts,
                  crowd_skipped=crowd_skipped, training_id_mapping={str(i): i-1 for i in range(1, 9)},
                  image_manifest_sha256=sha256(root / 'image_manifest.json'),
                  annotation_sha256=sha256(root / 'annotations_coco.json'),
                  scope='real train-only segmentation input; no query acceptance or performance evaluation')
    return records, report


def map_record(record, short_edge=256, max_size=384):
    """Deterministic small-check mapper; not the planned LSJ training recipe."""
    from detectron2.data import detection_utils as utils, transforms as T
    import torch
    record = copy.deepcopy(record)
    image = utils.read_image(record['file_name'], format='RGB')
    utils.check_image_size(record, image)
    transform = T.ResizeShortestEdge(short_edge, max_size).get_transform(image)
    image = transform.apply_image(image)
    shape = image.shape[:2]
    annotations = [utils.transform_instance_annotations(a, transform, shape)
                   for a in record.pop('annotations')]
    instances = utils.annotations_to_instances(annotations, shape, mask_format='bitmask')
    instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
    instances = utils.filter_empty_instances(instances)
    # Official Mask2Former prepare_targets expects a tensor, not BitMasks.
    instances.gt_masks = instances.gt_masks.tensor
    record['instances'] = instances
    record['image'] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
    return record
