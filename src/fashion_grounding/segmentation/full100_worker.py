"""Opt-in full100 GPU worker. Legacy trainers/evaluators remain unchanged.

Model, mapper, optimizer, scheduler, AMP replay, checkpoint state, AP inference and
COCO metrics reuse existing implementations. Heavy imports occur only in a worker.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import time

from .full100_contract import (cache_identity, digest_json, load_binding, read_json,
                               require, resolve, sha256)

ROOT = Path(__file__).resolve().parents[3]


class LossSummary:
    """Constant-memory aggregates; detailed rows are streamed to JSONL."""
    def __init__(self):
        self.count = self.samples = self.empty = 0
        self.first = self.last = None
        self.minimum, self.maximum = math.inf, -math.inf

    def add(self, row):
        loss = row['total_loss']
        require(math.isfinite(loss), 'Non-finite logged loss')
        if self.count == 0:
            self.first = loss
        self.count += 1
        self.last = loss
        self.minimum, self.maximum = min(self.minimum, loss), max(self.maximum, loss)
        self.samples += len(row['image_ids'])
        self.empty += sum(n == 0 for n in row['instance_counts'])

    def result(self):
        require(self.count > 0, 'No successful optimizer updates')
        return dict(first_loss=self.first, last_loss=self.last, min_loss=self.minimum,
                    max_loss=self.maximum, updates_this_process=self.count,
                    sample_count=self.samples, empty_augmented_images=self.empty)


def atomic_json(path, value):
    from .full100_runtime import atomic_json as write
    write(path, value)


def verified_contract(path):
    contract = read_json(path)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    require(not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'],
                                        cwd=ROOT, text=True).strip(), 'Worker source must be committed')
    actual = resolve(ROOT, contract['plan_path'], contract['inputs']['binding_path'], commit)
    require(actual == contract and contract['ready_for_training'] is True, 'Worker contract drift')
    categories = {c['id']: c['name'] for c in read_json(ROOT/contract['runtime']['categories']['path'])['categories']}
    _, metadata = load_binding(contract['inputs']['binding_path'], contract['runtime']['mapping']['sha256'],
                               categories, contract['inputs']['evaluation_data'])
    return contract, metadata


def training_records(package, manifest, coco, *, bbox_mode):
    indexed = {}
    for ann in sorted(coco['annotations'], key=lambda row: row['id']):
        if ann.get('iscrowd', 0):
            continue
        indexed.setdefault(ann['image_id'], []).append(dict(
            bbox=ann['bbox'], bbox_mode=bbox_mode, segmentation=ann['segmentation'],
            category_id=ann['category_id']-1, iscrowd=0))
    return [dict(file_name=str(Path(package)/im['file_name']), image_id=im['image_id'],
                 height=im['height'], width=im['width'], annotations=indexed.get(im['image_id'], []))
            for im in sorted(manifest['images'], key=lambda row: row['image_id'])]


def train(contract, metadata, output, stop, resume):
    import random
    import numpy as np
    import torch
    from detectron2.structures import BoxMode
    from .amp_recovery import AmpRecovery
    from .checkpoint import read_coco_checkpoint
    from .data import map_training_record
    from .model import adapt_coco_state, build_model
    from .train import configure_record, component_states, fingerprints
    from .training_state import (UpdateSchedule, StatefulSampler, build_optimizer, restore_rng,
                                 state_digest, save_checkpoint, load_checkpoint, checkpoint_due)

    config = contract['train_config']
    require(stop in [node['update'] for node in contract['schedule']['evaluation_nodes']], 'Invalid train boundary')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output/'resolved_config.json', config)
    atomic_json(output/'data_check.json', contract['inputs'])
    torch.set_num_threads(2)
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    records = training_records(contract['inputs']['train_package'], metadata[0], metadata[1],
                               bbox_mode=BoxMode.XYWH_ABS)
    sampler = StatefulSampler(len(records), config['seed'])
    model, cfg = build_model(contract['runtime']['upstream_source'], device='cpu')
    (output/'model_config.yaml').write_text(configure_record(cfg, config).dump())
    data = contract['inputs']['training_data']
    config_hash = contract['train_config_sha256']
    state = None
    if resume:
        state = load_checkpoint(resume, config_hash, data)
        require(state['git_commit'] == contract['source_commit'], 'Resume source differs')
        require(0 < state['completed_updates'] < stop, 'Invalid resume boundary')
        require(fingerprints({k: state[k] for k in state['component_digests']}) == state['component_digests'],
                'Stored component fingerprints differ')
        model.load_state_dict(state['model'], strict=True)
        initialization, initial_digests = state['initialization'], state['initial_parameter_digests']
    else:
        provenance = contract['upstream']
        weights, integrity = read_coco_checkpoint(contract['runtime']['initial_checkpoint'],
            expected_sha256=provenance['checkpoint_sha256'], expected_size=provenance['checkpoint_mirror']['bytes'],
            md5_prefix=provenance['checkpoint_md5_prefix'])
        initialization = dict(integrity=integrity, adaptation=adapt_coco_state(model, weights))
        del weights
        initial_digests = dict(backbone=state_digest(model.backbone.state_dict()),
                              classifier=state_digest(model.sem_seg_head.predictor.class_embed.state_dict()))
    model = model.to('cuda').train()
    optimizer = build_optimizer(model, config)
    scheduler = UpdateSchedule(optimizer, config)
    scaler = torch.cuda.amp.GradScaler(enabled=config['amp'], init_scale=config['amp_initial_scale'])
    completed, recovery_state = 0, None
    if state is not None:
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        sampler.load_state_dict(state['sampler'])
        restore_rng(state['rng'])
        actual = fingerprints(component_states(model, optimizer, scheduler, scaler, sampler))
        require(actual == state['component_digests'], 'Restored live state differs')
        completed, recovery_state = state['completed_updates'], state['amp_recovery_state']
        atomic_json(output/'resume_check.json', dict(status='passed', checkpoint_sha256=sha256(resume),
            completed_updates=completed, expected_digests=state['component_digests'], restored_digests=actual))
        del state
    start = completed
    atomic_json(output/'initialization.json', initialization)
    params = [p for p in model.parameters() if p.requires_grad]
    torch.cuda.reset_peak_memory_stats()

    def emit(event):
        with (output/'amp_overflow.jsonl').open('a') as log:
            log.write(json.dumps(event, allow_nan=False)+'\n')
    recovery = AmpRecovery(config['amp_recovery'], emit, recovery_state)

    def forward_backward():
        components, image_ids, counts = {}, [], []
        for _ in range(config['accumulation_steps']):
            batch = [map_training_record(records[sampler.next()], config) for _ in range(config['microbatch_images'])]
            image_ids.extend(row['image_id'] for row in batch)
            counts.extend(len(row['instances']) for row in batch)
            with torch.cuda.amp.autocast(enabled=config['amp']):
                losses = model(batch)
                loss = sum(losses.values()) / config['accumulation_steps']
            require(bool(losses) and torch.isfinite(loss).all()
                    and all(torch.isfinite(v).all() for v in losses.values()), 'Non-finite loss')
            scaler.scale(loss).backward()
            for key, value in losses.items():
                components[key] = components.get(key, 0.) + float(value.detach())/config['accumulation_steps']
            del batch, losses, loss
        return dict(loss_components=components, image_ids=image_ids, instance_counts=counts)

    summary, checkpoints = LossSummary(), []
    with (output/'metrics.jsonl').open('x') as log:
        while completed < stop:
            began = time.monotonic()
            details, retries = recovery.backward(model, optimizer, scaler, sampler,
                                                  forward_backward, completed+1, lambda: None)
            norm = torch.nn.utils.clip_grad_norm_(params, config['gradient_clip_norm'], error_if_nonfinite=True)
            require(norm > 0, 'Zero gradient')
            clipped = torch.linalg.vector_norm(torch.stack([p.grad.detach().float().norm()
                                                          for p in params if p.grad is not None]))
            require(clipped <= config['gradient_clip_norm']*1.001, 'Gradient clipping failed')
            scale = scaler.get_scale()
            used_lrs = sorted({g['lr'] for g in optimizer.param_groups})
            scaler.step(optimizer); scaler.update()
            require(scaler.get_scale() >= scale, 'Unexpected skipped optimizer update')
            scheduler.step(); completed += 1
            torch.cuda.synchronize()
            row = dict(update=completed, total_loss=sum(details['loss_components'].values()),
                       components=details['loss_components'], image_ids=details['image_ids'],
                       instance_counts=details['instance_counts'], sampler_epoch=sampler.epoch,
                       grad_norm_before_clip=float(norm), grad_norm_after_clip=float(clipped),
                       learning_rates=used_lrs, loss_scale=scale, amp_retries=retries,
                       amp_overflow_events_total=recovery.overflow_events, optimizer_step_skipped=False,
                       elapsed_seconds=time.monotonic()-began,
                       peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                       peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
            log.write(json.dumps(row, allow_nan=False)+'\n'); log.flush(); summary.add(row)
            if completed == start+1 or completed % 10 == 0:
                print(json.dumps({k: row[k] for k in ('update', 'total_loss', 'grad_norm_before_clip')}), flush=True)
            if checkpoint_due(config, completed, stop):
                require(all(torch.isfinite(p).all() for p in params), 'Non-finite checkpoint parameter')
                states = component_states(model, optimizer, scheduler, scaler, sampler)
                saved = dict(**states, component_digests=fingerprints(states), completed_updates=completed,
                             config=config, config_sha256=config_hash, data=data, git_commit=contract['source_commit'],
                             initialization=initialization, initial_parameter_digests=initial_digests,
                             resume_lineage=[], amp_recovery_state=recovery.state_dict())
                checkpoint = output/f'checkpoint_{completed:06d}.pth'
                digest = save_checkpoint(checkpoint, saved)
                stat = checkpoint.stat()
                receipt = dict(update=completed, path=str(checkpoint.resolve()), sha256=digest,
                               bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                               training_config_sha256=config_hash, source_commit=contract['source_commit'],
                               binding_sha256=contract['inputs']['binding_sha256'])
                atomic_json(checkpoint.with_suffix('.receipt.json'), receipt)
                checkpoints.append(receipt)
                del saved, states
    atomic_json(output/'train_report.json', dict(status='completed', start_update=start, completed_updates=completed,
        git_commit=contract['source_commit'], config_sha256=config_hash, **summary.result(),
        skipped_optimizer_updates=0, all_losses_and_gradients_finite=True,
        checkpoints=checkpoints, amp_recovery_state=recovery.state_dict(),
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated()))


def evaluate(contract, metadata, output, checkpoint):
    import numpy as np
    from PIL import Image
    import torch
    from .model import build_model
    from .evaluate import predict_ap
    from .evaluation import evaluate_masks
    from .training_state import load_checkpoint, state_digest

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    config = contract['eval_config']
    state = load_checkpoint(checkpoint, contract['train_config_sha256'], contract['inputs']['training_data'])
    require(state['git_commit'] == contract['source_commit'], 'Evaluation source lineage differs')
    expected_digest = state['component_digests']['model']
    require(state_digest(state['model']) == expected_digest, 'Checkpoint model fingerprint differs')
    item = dict(update=state['completed_updates'], sha256=sha256(checkpoint))
    model, cfg = build_model(contract['runtime']['upstream_source'], device='cpu')
    model.load_state_dict(state['model'], strict=True)
    del state
    model.to('cuda').eval()
    require(cfg.TEST.DETECTIONS_PER_IMAGE == 100, 'Wrong AP top-k')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    rows = []
    for im in sorted(metadata[2]['images'], key=lambda row: row['image_id']):
        try:
            with Image.open(Path(contract['inputs']['dev_package'])/im['file_name']) as image:
                pixels = np.asarray(image.convert('RGB'))
            detections = [dict(image_id=im['image_id'], **d) for d in predict_ap(pixels, model, config)]
            rows.append(dict(image_id=im['image_id'], status='ok' if detections else 'no_detections',
                             detections=detections))
        except Exception:
            rows.append(dict(image_id=im['image_id'], status='inference_error', detections=[],
                             error={'message': 'full100 inference failed'}))
            atomic_json(output/'predictions.json', rows)
            raise
    torch.cuda.synchronize()
    require(state_digest(model.state_dict()) == expected_digest, 'Evaluation changed model')
    atomic_json(output/'predictions.json', rows)
    result = evaluate_masks(metadata[3], rows)
    (output/'coco_summary.txt').write_text(result.pop('coco_summary'))
    report = dict(**result, **cache_identity(contract, item), purpose='dev_checkpoint_selection',
                  selection_eligible=result['complete'] and result['metrics']['AP'] is not None,
                  checkpoint_path=str(Path(checkpoint).resolve()),
                  predictions_sha256=sha256(output/'predictions.json'), optimizer_updates=0,
                  checkpoint_model_unchanged=True, queries_read=False,
                  peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
    atomic_json(output/'evaluation_report.json', report)
    (output/'evaluation_report.sha256').write_text(sha256(output/'evaluation_report.json')+'\n')
    atomic_json(output/'resolved_config.json', config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', required=True)
    parser.add_argument('--mode', choices=['train', 'evaluate'], required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--stop-update', type=int)
    parser.add_argument('--resume')
    parser.add_argument('--checkpoint')
    args = parser.parse_args()
    coordination = os.environ.get('FULL100_COORDINATION_RECORD')
    require(coordination is not None, 'Workers must be launched by the shared-lock full100 entry')
    owner = read_json(coordination)
    require(owner.get('entry') == 'full100' and owner.get('status') == 'running'
            and owner.get('wrapper_pid') == os.getppid(), 'Worker has no active owning launcher')
    contract, metadata = verified_contract(args.contract)
    require(owner['contract_sha256'] == digest_json(contract), 'Worker coordination contract differs')
    if args.mode == 'train':
        require(args.stop_update is not None and args.checkpoint is None, 'Invalid train arguments')
        train(contract, metadata, args.output, args.stop_update, args.resume)
    else:
        require(args.checkpoint is not None and args.resume is None and args.stop_update is None,
                'Invalid evaluation arguments')
        evaluate(contract, metadata, args.output, args.checkpoint)


if __name__ == '__main__':
    main()
