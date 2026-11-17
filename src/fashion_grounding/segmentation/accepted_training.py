"""Load already accepted records with metadata pins; reuse old pixel/geometry audit."""
from collections import Counter, defaultdict
import json
from pathlib import Path

from ..data.schema import safe_path, sha256
from .input_receipts import read_evidence_bundle, validate_chain


def load_accepted_training_package(root, evidence_path):
    from detectron2.structures import BoxMode
    bundle, evidence, _ = read_evidence_bundle(evidence_path)
    validate_chain(bundle, evidence)
    if bundle['mode'] != 'accepted_expansion':
        raise ValueError('Use existing loader for inputs without accepted expansion evidence')
    root = Path(root)
    data = dict(image_manifest_sha256=sha256(root/'image_manifest.json'),
                annotation_sha256=sha256(root/'annotations_coco.json'))
    if data != bundle['training_data'] or sha256(root/'SHA256SUMS.json') != evidence['acceptance']['package_pins']['SHA256SUMS.json']:
        raise ValueError('Accepted training metadata changed')
    manifest = json.loads((root/'image_manifest.json').read_text())
    coco = json.loads((root/'annotations_coco.json').read_text())
    if len(manifest['images']) != bundle['train_images'] or len(coco['annotations']) != bundle['train_instances']:
        raise ValueError('Accepted training population changed')
    annotations, counts, skipped = defaultdict(list), Counter(), 0
    for ann in sorted(coco['annotations'], key=lambda a:a['id']):
        if ann.get('iscrowd', 0):
            skipped += 1
            continue
        annotations[ann['image_id']].append(dict(segmentation=ann['segmentation'],
            bbox=ann['bbox'], bbox_mode=BoxMode.XYWH_ABS, category_id=ann['category_id']-1,
            annotation_id=ann['id'], source_area=ann['area'], source_category_name=ann.get('source_category_name'), iscrowd=0))
        counts[str(ann['category_id'])] += 1
    records = []
    for im in manifest['images']:
        if im['split'] != 'train' or im['source_dataset'] != 'fashionpedia':
            raise ValueError('Expected Fashionpedia train')
        records.append(dict(image_id=im['image_id'], file_name=str(safe_path(root, im['file_name'])),
                            height=im['height'], width=im['width'], annotations=annotations[im['image_id']]))
    report = dict(**data, images=len(records), instances=sum(counts.values()), crowd_skipped=skipped,
        project_category_counts=dict(counts), training_id_mapping={str(i):i-1 for i in range(1,9)},
        input_validation_bundle_sha256=sha256(evidence_path), image_and_mask_reaudit=False,
        source='unchanged immutable package; accepted expansion receipt reused', test_files_read=False)
    return records, report
