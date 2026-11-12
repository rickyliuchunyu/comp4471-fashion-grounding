"""Bounded five-image COCO initialization check; never constructs an optimizer."""
import argparse
import json
from pathlib import Path
import random
import shutil
import subprocess
import time

import numpy as np
from PIL import Image
from pycocotools import mask as mask_api
import torch

from ..data.schema import safe_path, sha256, write_json
from .checkpoint import read_coco_checkpoint
from .model import adapt_coco_state, build_model
from .predict import export_package, predict_instances


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('package', 'source', 'checkpoint', 'output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--provenance', default='envs/segmentation/upstream.json')
    parser.add_argument('--config', default='configs/runs/segmentation/coco_inference.json')
    args = parser.parse_args()
    output, package = Path(args.output), Path(args.package)
    config = json.loads(Path(args.config).read_text())
    provenance = json.loads(Path(args.provenance).read_text())
    if config['optimizer_updates'] != 0 or config['image_count'] != 5 or config['amp']:
        raise ValueError('Only the five-image FP32 inference check is permitted')
    manifest_path = package / 'image_manifest.json'
    if sha256(manifest_path) != config['source_manifest_sha256']:
        raise ValueError('Input manifest differs from the pinned pilot')
    manifest = json.loads(manifest_path.read_text())
    if manifest['schema_version'] != 3 or manifest['illustrative_only']:
        raise ValueError('Expected real schema-3 pilot images')
    selected = sorted(manifest['images'], key=lambda im: im['image_id'])[:config['image_count']]
    if len(selected) != 5 or len({im['image_id'] for im in selected}) != 5:
        raise ValueError('Need five unique input images')
    for im in selected:
        if im['source_dataset'] != 'fashionpedia' or im['split'] != 'train':
            raise ValueError('Only Fashionpedia train inputs permitted')
        if sha256(safe_path(package, im['file_name'])) != im['sha256']:
            raise ValueError('Input image hash mismatch')
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.set_num_threads(2)
    mirror = provenance['checkpoint_mirror']
    state, integrity = read_coco_checkpoint(args.checkpoint,
        expected_sha256=mirror['sha256'], expected_size=mirror['bytes'],
        md5_prefix=provenance['checkpoint_md5_prefix'])
    model, cfg = build_model(args.source, device='cpu')
    adaptation = adapt_coco_state(model, state)
    del state
    checkpoint = output / 'adapted_initialization.pth'
    torch.save(model.state_dict(), checkpoint)
    del model
    # Prove that the actual adapted artifact restores into a fresh model.
    restored = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model, _ = build_model(args.source, device='cpu')
    model.load_state_dict(restored, strict=True)
    for key, tensor in model.state_dict().items():
        if not torch.equal(tensor, restored[key]):
            raise ValueError(f'Adapted checkpoint restore mismatch: {key}')
    adaptation['adapted_checkpoint_strict_restore_verified'] = True
    write_json(output / 'load_report.json', dict(integrity=integrity, adaptation=adaptation))
    write_json(output / 'resolved_config.json', config)
    # Saved construction config is CPU; actual execution below is explicitly CUDA.
    (output / 'model_config.yaml').write_text(cfg.dump())
    model = model.to('cuda').eval()
    torch.cuda.reset_peak_memory_stats()
    predictions, image_reports = {}, []
    for im in selected:
        if time.monotonic()-started > config['budget_seconds']:
            raise TimeoutError('Short inference budget exceeded')
        with Image.open(safe_path(package, im['file_name'])) as opened:
            image = np.asarray(opened.convert('RGB'))
        if image.shape[:2] != (im['height'], im['width']):
            raise ValueError('Image dimensions differ from manifest')
        diagnostics = dict(image_id=im['image_id'], image_sha256=im['sha256'])
        torch.cuda.synchronize()
        inference_started = time.monotonic()
        predictions[im['image_id']] = predict_instances(image, model,
            {**config, 'image_id': im['image_id']}, diagnostics=diagnostics)
        torch.cuda.synchronize()
        diagnostics['elapsed_seconds'] = time.monotonic()-inference_started
        image_reports.append(diagnostics)
    for key, tensor in model.state_dict().items():
        if not torch.equal(tensor.detach().cpu(), restored[key]):
            raise ValueError(f'Inference modified model state: {key}')
    subset = output / 'input'
    subset.mkdir()
    manifest['manifest_id'] += '-coco-inference-five-images'
    manifest['images'] = selected
    for im in selected:
        dest = safe_path(subset, im['file_name'])
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(safe_path(package, im['file_name']), dest)
    write_json(subset / 'image_manifest.json', manifest)
    adapted_sha256 = sha256(checkpoint)
    candidates = export_package(output / 'candidate_package', subset / 'image_manifest.json',
        predictions, candidate_set_id=output.name+'-untrained-head', producer_run_id=output.name,
        checkpoint_sha256=adapted_sha256, decoder_config_sha256=sha256(output / 'resolved_config.json'),
        illustrative_only=True, note='COCO pretrained non-head tensors; randomly initialized eight-class head. Zero training updates. Not valid garment predictions or performance evidence.')
    roundtrips = 0
    for row in candidates['images']:
        for candidate, expected in zip(row['candidates'], predictions[row['image_id']]):
            path = safe_path(output / 'candidate_package', candidate['mask_reference'])
            if candidate['mask_sha256'] != sha256(path):
                raise ValueError('Exported mask hash mismatch')
            decoded = mask_api.decode(json.loads(path.read_text())).astype(bool)
            if not np.array_equal(decoded, expected.mask):
                raise ValueError('Original-size RLE roundtrip failed')
            roundtrips += 1
    report = dict(purpose=config['purpose'], status='completed', optimizer_updates=0,
        annotations_read=False, queries_read=False, execution_device=str(model.device),
        source_manifest_sha256=config['source_manifest_sha256'], images=image_reports,
        integrity=integrity, adaptation=adaptation, model_state_unchanged_after_inference=True,
        adapted_checkpoint_sha256=adapted_sha256, original_size_rle_roundtrips=roundtrips,
        candidate_manifest_sha256=sha256(output / 'candidate_package/candidates_predicted.json'),
        candidate_package_illustrative_only=True, performance_evaluated=False,
        elapsed_seconds=time.monotonic()-started,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
    write_json(output / 'inference_report.json', report)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
