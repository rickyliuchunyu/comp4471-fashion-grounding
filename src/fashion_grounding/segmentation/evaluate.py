"""Evaluate saved checkpoints with official instance inference; never train.

Run under scripts/with_gpu_lock.py. Pilot AP is an integration diagnostic and
cannot select a checkpoint. Formal dev requires separately frozen train/dev.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image
import torch

from ..data.schema import read_json, write_json, sha256, safe_path, encode_mask, require
from .evaluation import read_eval_config, load_eval_package, evaluate_masks
from .model import build_model
from .predict import predict_instances, export_package
from .training_state import config_sha256, state_digest


@torch.no_grad()
def predict_ap(image_rgb, model, config):
    """Only RGB pixels enter pinned upstream model.forward/instance_inference."""
    from detectron2.data import transforms as T
    require(not model.training, 'AP requires eval mode')
    require(model.instance_on and not model.panoptic_on and model.test_topk_per_image == 100,
            'Require upstream instance-only top-100 inference')
    h, w = image_rgb.shape[:2]
    resized = T.ResizeShortestEdge(config['short_edge'], config['max_size']).get_transform(image_rgb).apply_image(image_rgb)
    tensor = torch.as_tensor(np.ascontiguousarray(resized.transpose(2, 0, 1)), device=model.device)
    instances = model([dict(image=tensor, height=h, width=w)])[0]['instances'].to('cpu')
    require(torch.isfinite(instances.scores).all(), 'Non-finite AP scores')
    require(instances.pred_masks.shape[1:] == (h, w), 'AP mask must use original image dimensions')
    return [dict(category_id=int(label)+1, score=float(score), segmentation=encode_mask(mask.numpy().astype(bool)))
            for label, score, mask in zip(instances.pred_classes, instances.scores, instances.pred_masks)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('package', 'training-package', 'source', 'config', 'output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--checkpoints', nargs='+', required=True)
    parser.add_argument('--export-diagnostic-candidates', action='store_true')
    args = parser.parse_args()
    config = read_eval_config(args.config)
    diagnostic = config['purpose'] == 'pilot_ap_integration'
    require(not args.export_diagnostic_candidates or diagnostic, 'This export is only for diagnostic consumer checks')
    manifest, coco, data_check = load_eval_package(args.package, config, args.training_package)
    source_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    require(not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], text=True).strip(),
            'Commit tracked evaluation source before running')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'resolved_config.json', config)
    write_json(output/'data_check.json', data_check)
    started = time.monotonic()
    torch.set_num_threads(2)
    model, cfg = build_model(args.source, device='cpu')
    model.eval()
    # Keep actual upstream decoding; no language score threshold, argmax or NMS.
    require(cfg.TEST.DETECTIONS_PER_IMAGE == 100, 'Unexpected upstream AP top-k')
    (output/'model_config.yaml').write_text(cfg.dump())
    report_paths = []
    for checkpoint_name in args.checkpoints:
        checkpoint = Path(checkpoint_name).resolve()
        checkpoint_hash = sha256(checkpoint)
        require(checkpoint_hash == checkpoint.with_suffix('.sha256').read_text().strip(), 'Checkpoint hash mismatch')
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        require(state['data'] == data_check['training_data'], 'Checkpoint training data differs')
        require(config_sha256(state['config']) == state['config_sha256'], 'Training config fingerprint mismatch')
        require(state_digest(state['model']) == state['component_digests']['model'], 'Checkpoint model fingerprint mismatch')
        update = state['completed_updates']
        require(type(update) is int and update > 0 and state['scheduler']['completed_updates'] == update, 'Invalid checkpoint update')
        require(diagnostic or state['config']['purpose'] == 's0_training', 'Pilot checkpoint is ineligible for formal dev selection')
        training_metadata = dict(training_config_sha256=state['config_sha256'], training_git_commit=state['git_commit'])
        model.load_state_dict(state['model'], strict=True)
        expected_model_digest = state['component_digests']['model']
        del state
        model.to('cuda').eval()
        torch.cuda.reset_peak_memory_stats()
        run = output / f'update_{update:06d}'
        run.mkdir(exist_ok=False)
        rows, candidates = [], {}
        for im in sorted(manifest['images'], key=lambda im: im['image_id']):
            require(time.monotonic()-started < config['budget_seconds'], 'Evaluation time budget exceeded')
            with Image.open(safe_path(args.package, im['file_name'])) as image:
                pixels = np.asarray(image.convert('RGB'))
            try:
                detections = [dict(image_id=im['image_id'], **d) for d in predict_ap(pixels, model, config)]
                rows.append(dict(image_id=im['image_id'], status='ok' if detections else 'no_detections', detections=detections))
            except Exception as error:
                # Retain the image in the denominator; a failed run cannot select.
                rows.append(dict(image_id=im['image_id'], status='inference_error', detections=[],
                                 error=dict(code=type(error).__name__, message='AP inference failed; run is ineligible')))
                write_json(run/'predictions.json', rows)
                raise  # Preserve partial evidence, no completed report on execution failure.
            if args.export_diagnostic_candidates:
                candidates[im['image_id']] = predict_instances(pixels, model,
                    dict(short_edge=config['short_edge'], max_size=config['max_size'], image_id=im['image_id'],
                         score_threshold=0.5, max_candidates=100))
        torch.cuda.synchronize()
        require(state_digest(model.state_dict()) == expected_model_digest, 'Evaluation modified model state')
        write_json(run/'predictions.json', rows)
        result = evaluate_masks(coco, rows)
        (run/'coco_summary.txt').write_text(result.pop('coco_summary'))
        report = dict(**result, **training_metadata, schema_version=1, purpose=config['purpose'],
            selection_eligible=not diagnostic and result['complete'] and result['metrics']['AP'] is not None,
            checkpoint_path=str(checkpoint), checkpoint_sha256=checkpoint_hash, completed_updates=update,
            evaluation_git_commit=source_commit, evaluation_config_sha256=config_sha256(config),
            evaluation_data={k: config[k] for k in ('image_manifest_sha256', 'annotation_sha256')},
            training_data=data_check['training_data'], overlap=data_check['overlap'],
            predictions_sha256=sha256(run/'predictions.json'),
            optimizer_updates=0, checkpoint_model_unchanged=True, queries_read=False,
            elapsed_seconds=time.monotonic()-started, finished_at=datetime.now(timezone.utc).isoformat(),
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
        if args.export_diagnostic_candidates:
            export_package(run/'candidate_package', Path(args.package)/'image_manifest.json', candidates,
                candidate_set_id=output.name+f'-update-{update}', producer_run_id=output.name,
                checkpoint_sha256=checkpoint_hash,
                decoder_config_sha256=config_sha256(dict(decoder='language_one_class_per_query',
                    short_edge=config['short_edge'], max_size=config['max_size'], score_threshold=0.5, max_candidates=100)),
                illustrative_only=True, note='Train-pilot engineering checkpoint; no dev selection or production acceptance.')
            report['candidate_manifest_sha256'] = sha256(run/'candidate_package/candidates_predicted.json')
        write_json(run/'evaluation_report.json', report)
        (run/'evaluation_report.sha256').write_text(sha256(run/'evaluation_report.json')+'\n')
        report_paths.append(str(run/'evaluation_report.json'))
        print(json.dumps(dict(update=update, diagnostic=diagnostic, images=result['evaluated_images'],
                              metrics=result['metrics'], selection_eligible=report['selection_eligible'])), flush=True)
    write_json(output/'run_report.json', dict(status='completed', reports=report_paths,
        optimizer_updates=0, purpose=config['purpose'], elapsed_seconds=time.monotonic()-started))


if __name__ == '__main__':
    main()
