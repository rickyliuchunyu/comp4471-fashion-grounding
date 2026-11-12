"""Language candidates, deliberately separate from upstream COCO AP decoding."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch


@dataclass
class Candidate:
    candidate_id: str
    category_id: int
    score: float
    bbox_xyxy: tuple
    mask: np.ndarray
    decoder_query_id: int


def sha256(path):
    with Path(path).open('rb') as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')


def safe_path(root, name):
    relative = Path(name)
    if relative.is_absolute() or '..' in relative.parts or ':' in name or '\\' in name:
        raise ValueError('Expected package-relative file path')
    return Path(root) / relative


def decode_candidates(class_logits, mask_logits, image_id, score_threshold=0.5,
                      max_candidates=100):
    """Input Qx9 logits and QxHxW logits already restored to original pixels.

    Softmax includes no-object; argmax considers the eight foreground classes.
    Score = foreground class probability * sigmoid mean on logit>0 pixels,
    including upstream's 1e-6 denominator epsilon. No NMS or GT input.
    """
    if type(image_id) is not int or image_id <= 0:
        raise ValueError('image_id must be a positive integer')
    if not 0 <= score_threshold <= 1 or type(max_candidates) is not int or not 1 <= max_candidates <= 100:
        raise ValueError('Invalid candidate decoder settings')
    if class_logits.ndim != 2 or class_logits.shape[1] != 9:
        raise ValueError('Expected Qx9 class logits')
    if mask_logits.ndim != 3 or mask_logits.shape[0] != class_logits.shape[0] or min(mask_logits.shape[1:]) <= 0:
        raise ValueError('Expected matching QxHxW mask logits')
    if not torch.isfinite(class_logits).all() or not torch.isfinite(mask_logits).all():
        raise ValueError('Non-finite decoder output')
    probabilities = class_logits.float().softmax(-1)[:, :8]
    scores, labels = probabilities.max(-1)
    masks = mask_logits > 0
    quality = (mask_logits.float().sigmoid() * masks).flatten(1).sum(1)
    areas = masks.flatten(1).sum(1)
    scores = scores * quality / (areas + 1e-6)
    candidates = []
    for qid in range(len(scores)):
        if areas[qid] == 0 or scores[qid] < score_threshold:
            continue
        mask = masks[qid].detach().cpu().numpy().astype(bool)
        ys, xs = np.nonzero(mask)
        candidates.append(Candidate(
            f'image-{image_id}-query-{qid:03d}', int(labels[qid]) + 1,
            float(scores[qid]), (int(xs.min()), int(ys.min()), int(xs.max()) + 1,
                                 int(ys.max()) + 1), mask, qid))
    return sorted(candidates, key=lambda c: (-c.score, c.candidate_id))[:max_candidates]


@torch.no_grad()
def predict_instances(image_rgb, model, config):
    """Only pixels + image ID/settings enter inference; annotations are not accepted.

    Uses upstream preprocessing/backbone/head and sem_seg_postprocess for padding
    removal and original-size bilinear resize before mask thresholding.
    """
    from detectron2.structures import ImageList
    from detectron2.modeling.postprocessing import sem_seg_postprocess
    from detectron2.data import transforms as T
    if image_rgb.dtype != np.uint8 or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError('Expected uint8 RGB HxWx3')
    if model.training:
        raise ValueError('Inference requires model.eval()')
    h, w = image_rgb.shape[:2]
    resized = T.ResizeShortestEdge(config['short_edge'], config['max_size']).get_transform(image_rgb).apply_image(image_rgb)
    tensor = torch.as_tensor(np.ascontiguousarray(resized.transpose(2, 0, 1)), device=model.device).float()
    images = ImageList.from_tensors([(tensor - model.pixel_mean) / model.pixel_std], model.size_divisibility)
    outputs = model.sem_seg_head(model.backbone(images.tensor))
    masks = torch.nn.functional.interpolate(outputs['pred_masks'], size=images.tensor.shape[-2:],
                                           mode='bilinear', align_corners=False)[0]
    masks = sem_seg_postprocess(masks, images.image_sizes[0], h, w)
    return decode_candidates(outputs['pred_logits'][0], masks, config['image_id'],
                             config['score_threshold'], config['max_candidates'])


def export_package(output, image_manifest, predictions, *, candidate_set_id,
                   producer_run_id, checkpoint_sha256, decoder_config_sha256,
                   illustrative_only=False, note=''):
    """Write a NEW self-contained package; failures and empty results stay distinct.

    predictions maps every manifest image ID to candidates or {code,message}.
    Error messages must be sanitized by the caller. Masks use COCO compressed RLE.
    """
    from pycocotools import mask as mask_api
    output, image_manifest = Path(output), Path(image_manifest)
    manifest = json.loads(image_manifest.read_text(encoding='utf-8'))
    if manifest['schema_version'] != 3 or not candidate_set_id or not producer_run_id:
        raise ValueError('Invalid schema or provenance IDs')
    if manifest['illustrative_only'] and not illustrative_only:
        raise ValueError('Cannot relabel synthetic input as real')
    for digest in (checkpoint_sha256, decoder_config_sha256):
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Expected checkpoint/config SHA256')
    ids = [im['image_id'] for im in manifest['images']]
    if len(ids) != len(set(ids)) or set(predictions) != set(ids):
        raise ValueError('Predictions must cover each manifest image exactly once')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'masks').mkdir()
    # Preserve the manifest bytes/hash; copy its images without modifying upstream.
    for im in manifest['images']:
        src = safe_path(image_manifest.parent, im['file_name'])
        if sha256(src) != im['sha256']:
            raise ValueError('Image hash mismatch')
        dest = safe_path(output, im['file_name'])
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
    shutil.copyfile(image_manifest, output / 'image_manifest.json')
    rows, seen = [], set()
    for im in manifest['images']:
        result = predictions[im['image_id']]
        row = dict(image_id=im['image_id'], image_size_wh=[im['width'], im['height']], candidates=[])
        if isinstance(result, dict):
            if set(result) != {'code', 'message'} or not all(isinstance(v, str) and v for v in result.values()):
                raise ValueError('Error requires nonempty code/message')
            row.update(status='inference_error', error=result)
        else:
            row['status'] = 'ok' if result else 'no_candidates'
            if result != sorted(result, key=lambda c: (-c.score, c.candidate_id)):
                raise ValueError('Candidate order mismatch')
            decoder_ids = set()
            for candidate in result:
                if candidate.candidate_id in seen or candidate.decoder_query_id in decoder_ids:
                    raise ValueError('Duplicate candidate or decoder query ID')
                seen.add(candidate.candidate_id)
                decoder_ids.add(candidate.decoder_query_id)
                mask = candidate.mask
                if mask.dtype != bool or mask.shape != (im['height'], im['width']) or not mask.any():
                    raise ValueError('Invalid original-size mask')
                ys, xs = np.nonzero(mask)
                if tuple(candidate.bbox_xyxy) != (xs.min(), ys.min(), xs.max()+1, ys.max()+1):
                    raise ValueError('bbox must be derived from predicted mask')
                if not 1 <= candidate.category_id <= 8 or not np.isfinite(candidate.score):
                    raise ValueError('Invalid category/score')
                rle = mask_api.encode(np.asfortranarray(mask, dtype=np.uint8))
                rle['counts'] = rle['counts'].decode('ascii')
                relative = f'masks/{im["image_id"]}-{candidate.decoder_query_id}.json'
                write_json(output / relative, rle)
                row['candidates'].append(dict(candidate_id=candidate.candidate_id,
                    category_id=candidate.category_id, score=candidate.score,
                    bbox_xyxy=list(candidate.bbox_xyxy), mask_reference=relative,
                    mask_sha256=sha256(output / relative), decoder_query_id=candidate.decoder_query_id,
                    source_annotation_id=None))
        rows.append(row)
    package = dict(schema_version=3, illustrative_only=illustrative_only,
        candidate_set_id=candidate_set_id, candidate_source='predicted',
        image_manifest=dict(path='image_manifest.json', sha256=sha256(image_manifest)),
        provenance=dict(producer_run_id=producer_run_id, checkpoint_sha256=checkpoint_sha256,
                        decoder_config_sha256=decoder_config_sha256,
                        annotation_manifest_sha256=None, note=note), images=rows)
    write_json(output / 'candidates_predicted.json', package)
    return package
