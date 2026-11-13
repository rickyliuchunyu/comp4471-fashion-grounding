"""Single-GPU Mask2Former training with bounded runs and complete resume state.

No validation/model selection is performed here. Run under with_gpu_lock.py.
The pilot profile exercises this same trainer for 200 updates only.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import shutil
import subprocess
import time

import numpy as np
import torch

from ..data.schema import sha256, write_json
from .checkpoint import read_coco_checkpoint
from .data import load_training_package, map_training_record
from .model import adapt_coco_state, build_model
from .resume_extension import load_extended_checkpoint
from .training_state import (UpdateSchedule, StatefulSampler, build_optimizer, capture_rng,
    restore_rng, state_digest, save_checkpoint, load_checkpoint, read_config, config_sha256, checkpoint_due)


def component_states(model, optimizer, scheduler, scaler, sampler):
    return dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                sampler=sampler.state_dict(), rng=capture_rng())


def fingerprints(states):
    return {key: state_digest(value) for key, value in states.items()}


def configure_record(cfg, config):
    """Record the actual project solver/mapper choices, not the inherited COCO budget."""
    cfg = cfg.clone()
    cfg.defrost()
    cfg.MODEL.DEVICE = 'cuda'
    cfg.SOLVER.BASE_LR = config['learning_rate']
    cfg.SOLVER.BACKBONE_MULTIPLIER = config['backbone_lr_multiplier']
    cfg.SOLVER.WEIGHT_DECAY = config['weight_decay']
    cfg.SOLVER.WEIGHT_DECAY_NORM = config['weight_decay_norm']
    cfg.SOLVER.WEIGHT_DECAY_EMBED = config['weight_decay_embed']
    cfg.SOLVER.IMS_PER_BATCH = config['effective_batch_images']
    cfg.SOLVER.MAX_ITER = config['max_updates']
    cfg.SOLVER.STEPS = tuple(config['lr_milestones'])
    cfg.SOLVER.GAMMA = config['lr_decay']
    cfg.SOLVER.WARMUP_ITERS = config['warmup_updates']
    cfg.SOLVER.WARMUP_FACTOR = config['warmup_start_factor']
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = config['gradient_clip_norm']
    cfg.SOLVER.AMP.ENABLED = config['amp']
    cfg.INPUT.IMAGE_SIZE = config['image_size']
    cfg.INPUT.MIN_SCALE, cfg.INPUT.MAX_SCALE = config['min_scale'], config['max_scale']
    cfg.INPUT.RANDOM_FLIP = config['random_flip']
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.TEST.EVAL_PERIOD = 0
    # Data is consumed directly from the pinned package, never the COCO catalog.
    cfg.DATASETS.TRAIN, cfg.DATASETS.TEST = (), ()
    cfg.freeze()
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('package', 'source', 'output', 'config'):
        parser.add_argument('--'+name, required=True)
    initial = parser.add_mutually_exclusive_group(required=True)
    initial.add_argument('--checkpoint')
    initial.add_argument('--resume')
    parser.add_argument('--resume-extension', help='Reviewed budget-only transition JSON; requires --resume')
    parser.add_argument('--provenance', default='envs/segmentation/upstream.json')
    parser.add_argument('--stop-after', type=int, help='Absolute update boundary; never exceeds config max_updates')
    args = parser.parse_args()
    if args.resume_extension and not args.resume:
        parser.error('--resume-extension requires --resume')
    config = read_config(args.config)
    stop = config['max_updates'] if args.stop_after is None else args.stop_after
    if not 0 < stop <= config['max_updates']:
        raise ValueError('Stop boundary outside configured budget')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    if shutil.disk_usage(output).free < 3 * 1024**3:
        raise OSError('Need at least 3 GiB free for bounded checkpoint writes')
    torch.set_num_threads(2)
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    records, data_report = load_training_package(args.package)
    data = {key: data_report[key] for key in ('image_manifest_sha256', 'annotation_sha256')}
    if any(config[key] != data[key] for key in data):
        raise ValueError('Training input differs from pinned data hashes')
    records.sort(key=lambda r: r['image_id'])
    sampler = StatefulSampler(len(records), config['seed'])
    model, cfg = build_model(args.source, device='cpu')
    cfg = configure_record(cfg, config)
    digest = config_sha256(config)
    source_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], text=True).strip():
        raise ValueError('Commit tracked source changes before training')
    resume_state = None
    resume_lineage = []
    if args.resume:
        if args.resume_extension:
            repo_root = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], text=True).strip()
            resume_state, extension = load_extended_checkpoint(
                args.resume, config, data, args.resume_extension, source_commit, repo_root)
            resume_lineage = list(resume_state.get('resume_lineage', [])) + [extension]
            write_json(output / 'budget_extension.json', extension)
        else:
            resume_state = load_checkpoint(args.resume, digest, data)
            if resume_state['git_commit'] != source_commit:
                raise ValueError('Resume requires the same recorded source commit')
            resume_lineage = list(resume_state.get('resume_lineage', []))
        model.load_state_dict(resume_state['model'], strict=True)
        initialization = resume_state['initialization']
        initial_digests = resume_state['initial_parameter_digests']
    else:
        provenance = json.loads(Path(args.provenance).read_text())
        mirror = provenance['checkpoint_mirror']
        state, integrity = read_coco_checkpoint(args.checkpoint, expected_sha256=mirror['sha256'],
            expected_size=mirror['bytes'], md5_prefix=provenance['checkpoint_md5_prefix'])
        adaptation = adapt_coco_state(model, state)
        initialization = dict(integrity=integrity, adaptation=adaptation)
        del state
        initial_digests = dict(backbone=state_digest(model.backbone.state_dict()),
                               classifier=state_digest(model.sem_seg_head.predictor.class_embed.state_dict()))
    model = model.to('cuda').train()
    optimizer = build_optimizer(model, config)
    scheduler = UpdateSchedule(optimizer, config)
    scaler = torch.cuda.amp.GradScaler(enabled=config['amp'], init_scale=config['amp_initial_scale'])
    completed = 0
    if resume_state is not None:
        expected = resume_state['component_digests']
        # Verify stored fingerprints, then verify freshly restored live components.
        if fingerprints({k: resume_state[k] for k in expected}) != expected:
            raise ValueError('Checkpoint component fingerprint mismatch')
        optimizer.load_state_dict(resume_state['optimizer'])
        scheduler.load_state_dict(resume_state['scheduler'])
        scaler.load_state_dict(resume_state['scaler'])
        sampler.load_state_dict(resume_state['sampler'])
        restore_rng(resume_state['rng'])
        actual = fingerprints(component_states(model, optimizer, scheduler, scaler, sampler))
        if actual != expected:
            raise ValueError(f'Restored state mismatch: {[k for k in expected if actual[k] != expected[k]]}')
        completed = resume_state['completed_updates']
        write_json(output / 'resume_check.json', dict(status='passed', resumed_from_update=completed,
            checkpoint_sha256=sha256(args.resume), exact_components=sorted(expected),
            expected_digests=expected, restored_digests=actual))
        del resume_state
    if completed >= stop:
        raise ValueError('Checkpoint already reaches the requested stop boundary')
    start_update = completed
    write_json(output / 'resolved_config.json', config)
    write_json(output / 'data_check.json', data_report)
    write_json(output / 'initialization.json', initialization)
    (output / 'model_config.yaml').write_text(cfg.dump())
    write_json(output / 'optimizer_groups.json', [dict(name=g['parameter_name'], lr=g['lr'],
        weight_decay=g['weight_decay'], shape=list(g['params'][0].shape)) for g in optimizer.param_groups])
    torch.cuda.reset_peak_memory_stats()
    params = [p for p in model.parameters() if p.requires_grad]
    loss_rows, checkpoints = [], []
    with (output / 'metrics.jsonl').open('x') as log:
        while completed < stop:
            if time.monotonic()-started > config['budget_seconds']:
                raise TimeoutError('Configured wall-time budget exceeded')
            update_started = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            components, image_ids, instance_counts = {}, [], []
            for _ in range(config['accumulation_steps']):
                batch = [map_training_record(records[sampler.next()], config) for _ in range(config['microbatch_images'])]
                image_ids.extend(item['image_id'] for item in batch)
                instance_counts.extend(len(item['instances']) for item in batch)
                with torch.cuda.amp.autocast(enabled=config['amp']):
                    losses = model(batch)
                    loss = sum(losses.values()) / config['accumulation_steps']
                if not losses or any(not torch.isfinite(v).all() for v in losses.values()):
                    raise ValueError('Non-finite or missing loss')
                scaler.scale(loss).backward()
                for key, value in losses.items():
                    components[key] = components.get(key, 0.) + float(value.detach()) / config['accumulation_steps']
                del batch, losses, loss
            scaler.unscale_(optimizer)
            bad_gradients = [name for name, param in model.named_parameters()
                             if param.grad is not None and not torch.isfinite(param.grad).all()]
            if bad_gradients:
                write_json(output / 'gradient_failure.json', dict(attempted_update=completed+1,
                    completed_updates=completed, image_ids=image_ids, instance_counts=instance_counts,
                    loss_components=components, loss_scale=scaler.get_scale(),
                    nonfinite_gradient_parameters=bad_gradients,
                    peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated()))
                raise ValueError('Non-finite gradients; no optimizer/scheduler step performed')
            grad_norm = torch.nn.utils.clip_grad_norm_(params, config['gradient_clip_norm'], error_if_nonfinite=True)
            if grad_norm <= 0:
                raise ValueError('Zero full-model gradient')
            clipped_norm = torch.linalg.vector_norm(torch.stack([p.grad.detach().float().norm() for p in params if p.grad is not None]))
            if clipped_norm > config['gradient_clip_norm'] * 1.001:
                raise ValueError('Full-model gradient clipping failed')
            scale_before = scaler.get_scale()
            used_lrs = sorted(set(g['lr'] for g in optimizer.param_groups))
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < scale_before:
                raise ValueError('AMP overflow skipped an optimizer update')
            scheduler.step()
            completed += 1
            torch.cuda.synchronize()
            row = dict(update=completed, total_loss=sum(components.values()), components=components,
                grad_norm_before_clip=float(grad_norm), grad_norm_after_clip=float(clipped_norm),
                loss_scale=scale_before, optimizer_step_skipped=False, learning_rates=used_lrs,
                image_ids=image_ids, instance_counts=instance_counts, sampler_epoch=sampler.epoch,
                elapsed_seconds=time.monotonic()-update_started,
                cuda_allocated_bytes=torch.cuda.memory_allocated(), cuda_reserved_bytes=torch.cuda.memory_reserved(),
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
            log.write(json.dumps(row, allow_nan=False)+'\n'); log.flush()
            loss_rows.append(row)
            if completed == start_update+1 or completed % 10 == 0:
                print(json.dumps({k: row[k] for k in ('update', 'total_loss', 'grad_norm_before_clip',
                    'loss_scale', 'peak_cuda_allocated_bytes')}), flush=True)
            if checkpoint_due(config, completed, stop):
                if any(not torch.isfinite(p).all() for p in params):
                    raise ValueError('Non-finite model parameters at checkpoint')
                states = component_states(model, optimizer, scheduler, scaler, sampler)
                state = dict(**states, component_digests=fingerprints(states), completed_updates=completed,
                    config=config, config_sha256=digest, data=data, git_commit=source_commit,
                    initialization=initialization, initial_parameter_digests=initial_digests,
                    resume_lineage=resume_lineage)
                path = output / f'checkpoint_{completed:06d}.pth'
                checkpoint_hash = save_checkpoint(path, state)
                checkpoints.append(dict(update=completed, path=path.name, sha256=checkpoint_hash))
                del state, states
    final_digests = dict(backbone=state_digest(model.backbone.state_dict()),
                        classifier=state_digest(model.sem_seg_head.predictor.class_embed.state_dict()))
    if any(final_digests[k] == initial_digests[k] for k in final_digests):
        raise ValueError('Expected both backbone and classifier to update')
    report = dict(status='completed', purpose=config['purpose'], git_commit=source_commit,
        config_sha256=digest, start_update=start_update, completed_updates=completed,
        updates_this_process=completed-start_update, resumed=bool(args.resume), resume_lineage=resume_lineage,
        elapsed_seconds=time.monotonic()-started, finished_at=datetime.now(timezone.utc).isoformat(),
        first_loss=loss_rows[0]['total_loss'], last_loss=loss_rows[-1]['total_loss'],
        min_loss=min(r['total_loss'] for r in loss_rows), max_loss=max(r['total_loss'] for r in loss_rows),
        all_losses_and_gradients_finite=True, skipped_optimizer_updates=0,
        initial_parameter_digests=initial_digests, final_parameter_digests=final_digests,
        backbone_and_classifier_updated=True, checkpoints=checkpoints,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(), performance_evaluated=False,
        sample_count=sum(len(r['image_ids']) for r in loss_rows),
        empty_augmented_images=sum(n == 0 for r in loss_rows for n in r['instance_counts']))
    write_json(output / 'train_report.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
