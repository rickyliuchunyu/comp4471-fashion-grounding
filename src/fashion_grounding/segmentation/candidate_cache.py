"""Streaming, image-only candidate cache and model-free schema-3 export.

The cache is after fixed one-class-per-query/mask decoding, BEFORE score filtering.
It is not a logit cache: changing preprocessing, class/mask decoding or NMS needs
a new forward pass. Empty masks are discarded, never replaced by GT geometry.
"""
import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np

from ..data.schema import (read_json, write_json, sha256, safe_path, require,
                           encode_mask, decode_mask, mask_box)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def digest(value):
    require(isinstance(value, str) and len(value) == 64 and
            all(c in '0123456789abcdef' for c in value), 'Invalid SHA256')
    return value


def image_index(manifest):
    require(manifest['schema_version'] == 3 and type(manifest['illustrative_only']) is bool,
            'Expected schema-3 image manifest')
    images = manifest['images']
    ids = [im['image_id'] for im in images]
    require(images and all(type(i) is int and i > 0 for i in ids) and
            len(set(ids)) == len(ids), 'Duplicate/invalid image IDs')
    for im in images:
        require(im['split'] in ('train', 'dev'), 'Only train/dev input is allowed')
        require(all(type(im[k]) is int and im[k] > 0 for k in ('width', 'height')),
                'Invalid image dimensions')
        digest(im['sha256'])
    return {im['image_id']: im for im in images}


def validate_row(row, im, root):
    require(row['image_id'] == im['image_id'] and
            row['image_size_wh'] == [im['width'], im['height']], 'Wrong image association')
    cs = row['candidates']
    require(row['decoder_query_count'] == 100 and row['discarded_empty_masks'] == 100-len(cs),
            'Cache must cover all 100 queries before score filtering')
    require(row['status'] == ('ok' if cs else 'no_candidates'), 'Incomplete/failed image')
    seen = set()
    for c in cs:
        qid = c['decoder_query_id']
        require(type(qid) is int and 0 <= qid < 100 and qid not in seen, 'Invalid/duplicate query ID')
        seen.add(qid)
        require(c['candidate_id'] == f'image-{im["image_id"]}-query-{qid:03d}', 'Unstable candidate ID')
        require(type(c['category_id']) is int and 1 <= c['category_id'] <= 8, 'Invalid category')
        require(type(c['score']) in (int, float) and math.isfinite(c['score']) and
                0 <= c['score'] <= 1, 'Invalid score')
        require(c['source_annotation_id'] is None, 'GT association forbidden')
        path = safe_path(root, c['mask_reference'])
        require(sha256(path) == digest(c['mask_sha256']), 'Mask hash mismatch')
        mask = decode_mask(read_json(path), im['height'], im['width'])
        require(c['bbox_xyxy'] == mask_box(mask), 'Mask/box mismatch')
    require(cs == sorted(cs, key=lambda c: (-c['score'], c['candidate_id'])), 'Unstable candidate order')


class CacheWriter:
    """Write one image at a time; cache.json exists only after complete coverage."""
    def __init__(self, output, image_manifest, provenance, decoder_config):
        self.root = Path(output)
        self.source = Path(image_manifest)
        self.manifest = read_json(self.source)
        self.images = image_index(self.manifest)
        require(provenance['producer_run_id'], 'Missing producer run ID')
        digest(provenance['checkpoint_sha256'])
        require(decoder_config['score_threshold'] == 0 and decoder_config['max_candidates'] == 100,
                'Cache cannot discard low-score queries')
        self.provenance = dict(provenance, source_image_manifest_sha256=sha256(self.source),
                               decoder_config_sha256=fingerprint(decoder_config),
                               annotation_manifest_sha256=None)
        self.root.mkdir(parents=True, exist_ok=False)
        write_json(self.root/'resolved_decoder_config.json', decoder_config)
        write_json(self.root/'run_metadata.json', dict(status='running', completed_images=0,
                   provenance=self.provenance))
        self.records = {}
        self.copied_images = {}

    def append(self, image_id, candidates, diagnostics):
        require(image_id in self.images and image_id not in self.records, 'Unknown/duplicate image')
        im = self.images[image_id]
        require(diagnostics['class_logits_shape'] == [100, 9] and
                diagnostics['mask_logits_original_shape'] == [100, im['height'], im['width']] and
                diagnostics['all_logits_finite'] is True and
                diagnostics['candidate_count'] == len(candidates), 'Incomplete decoder diagnostics')
        src = safe_path(self.source.parent, im['file_name'])
        require(sha256(src) == im['sha256'], 'Image hash mismatch')
        # Canonical names avoid collisions with metadata or caller-supplied paths.
        relative = f'images/{image_id}{src.suffix.lower()}'
        dest = safe_path(self.root, relative)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        row = dict(image_id=image_id, image_size_wh=[im['width'], im['height']],
                   status='ok' if candidates else 'no_candidates', decoder_query_count=100,
                   discarded_empty_masks=100-len(candidates), candidates=[])
        seen = set()
        for c in candidates:
            qid = c.decoder_query_id
            require(type(qid) is int and 0 <= qid < 100 and qid not in seen, 'Invalid/duplicate query ID')
            seen.add(qid)
            require(c.mask.dtype == bool and c.mask.shape == (im['height'], im['width']) and
                    c.mask.any(), 'Invalid original-size mask')
            ref = f'masks/{image_id}-{qid}.json'
            write_json(self.root/ref, encode_mask(c.mask))
            row['candidates'].append(dict(candidate_id=c.candidate_id, decoder_query_id=qid,
                category_id=c.category_id, score=c.score, bbox_xyxy=list(c.bbox_xyxy),
                mask_reference=ref, mask_sha256=sha256(self.root/ref), source_annotation_id=None))
        validate_row(row, im, self.root)
        record = f'records/{image_id}.json'
        write_json(self.root/record, row)
        self.records[image_id] = dict(image_id=image_id, path=record, sha256=sha256(self.root/record))
        self.copied_images[image_id] = dict(im, file_name=relative)

    def finish(self):
        require(set(self.records) == set(self.images), 'Incomplete cache; missing images')
        require(not (self.root/'cache.json').exists(), 'Cache already finalized')
        manifest = dict(self.manifest, images=[self.copied_images[i] for i in sorted(self.images)])
        write_json(self.root/'image_manifest.json', manifest)
        cache = dict(cache_schema_version=1, status='complete',
            illustrative_only=manifest['illustrative_only'], provenance=self.provenance,
            image_manifest=dict(path='image_manifest.json', sha256=sha256(self.root/'image_manifest.json')),
            decoder_config=dict(path='resolved_decoder_config.json', sha256=sha256(self.root/'resolved_decoder_config.json')),
            records=[self.records[i] for i in sorted(self.images)])
        write_json(self.root/'cache.json', cache)
        (self.root/'cache.sha256').write_text(sha256(self.root/'cache.json')+'\n')
        write_json(self.root/'run_metadata.json', dict(status='complete', completed_images=len(self.images),
                   cache_sha256=sha256(self.root/'cache.json'), provenance=self.provenance))
        return cache


def export_cached(cache_root, output, *, threshold, candidate_set_id, expected_checkpoint_sha256):
    """CPU-only projection; no PyTorch, checkpoint deserialization or model import.

    Threshold trials are NOT a frozen threshold selection or a benchmark result.
    """
    require(type(threshold) in (int, float) and math.isfinite(threshold) and
            threshold in (0.3, 0.5, 0.7), 'Use the registered threshold grid 0.3/0.5/0.7')
    require(isinstance(candidate_set_id, str) and candidate_set_id.strip(), 'Missing candidate set ID')
    root, output = Path(cache_root), Path(output)
    cache_hash = sha256(root/'cache.json')
    require(cache_hash == (root/'cache.sha256').read_text().strip(), 'Cache hash mismatch')
    cache = read_json(root/'cache.json')
    require(cache['cache_schema_version'] == 1 and cache['status'] == 'complete', 'Incomplete cache')
    p = cache['provenance']
    require(p['checkpoint_sha256'] == digest(expected_checkpoint_sha256), 'Wrong checkpoint cache')
    require(p['producer_run_id'] and p['annotation_manifest_sha256'] is None, 'Invalid provenance')
    digest(p['source_image_manifest_sha256'])
    def referenced(ref):
        path = safe_path(root, ref['path'])
        require(sha256(path) == digest(ref['sha256']), 'Referenced file hash mismatch')
        return path
    manifest_path = referenced(cache['image_manifest'])
    manifest = read_json(manifest_path)
    images = image_index(manifest)
    require(cache['illustrative_only'] == manifest['illustrative_only'], 'Illustrative flag mismatch')
    decoder = read_json(referenced(cache['decoder_config']))
    require(fingerprint(decoder) == p['decoder_config_sha256'] and
            decoder['score_threshold'] == 0 and decoder['max_candidates'] == 100,
            'Wrong pre-threshold decoder config')
    records = cache['records']
    require([r['image_id'] for r in records] == sorted(images), 'Incomplete/duplicate cache records')
    output.mkdir(parents=True, exist_ok=False)
    # Failures leave no candidates_predicted.json: partial output is not consumable.
    write_json(output/'run_metadata.json', dict(status='running', source_cache_sha256=cache_hash))
    shutil.copyfile(manifest_path, output/'image_manifest.json')
    rows = []
    for ref in records:
        im = images[ref['image_id']]
        row = read_json(referenced(ref))
        validate_row(row, im, root)
        src = safe_path(root, im['file_name'])
        require(sha256(src) == im['sha256'], 'Image hash mismatch')
        dest = safe_path(output, im['file_name'])
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        # Existing decoder compares a float32 tensor with a scalar: PyTorch
        # rounds the threshold to float32 (not Python float64, notably at .7).
        cutoff = float(np.float32(threshold))
        cs = [c for c in row['candidates'] if c['score'] >= cutoff]
        for c in cs:
            dest = safe_path(output, c['mask_reference'])
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(safe_path(root, c['mask_reference']), dest)
        rows.append(dict(image_id=im['image_id'], image_size_wh=row['image_size_wh'],
                         status='ok' if cs else 'no_candidates', candidates=cs))
    resolved = dict(decoder, score_threshold=threshold)
    write_json(output/'resolved_decoder_config.json', resolved)
    package = dict(schema_version=3, illustrative_only=manifest['illustrative_only'],
        candidate_source='predicted', candidate_set_id=candidate_set_id,
        image_manifest=dict(path='image_manifest.json', sha256=sha256(output/'image_manifest.json')),
        provenance=dict(p, decoder_config_sha256=fingerprint(resolved), source_cache_sha256=cache_hash,
                        score_threshold=threshold, threshold_status='trial_unselected',
                        effective_score_cutoff=float(np.float32(threshold)),
                        note='Offline threshold trial; accepted language-dev selection and receiver acceptance pending.'),
        images=rows)
    write_json(output/'run_metadata.json', dict(status='complete', source_cache_sha256=cache_hash,
        model_inference_calls=0, checkpoint_loads=0, optimizer_updates=0,
        images=len(rows), candidates=sum(len(r['candidates']) for r in rows),
        threshold_status='trial_unselected', selected_threshold=None))
    write_json(output/'candidates_predicted.json', package)
    return package
