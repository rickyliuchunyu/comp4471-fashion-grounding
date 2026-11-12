"""Bounded engineering check: real train inputs, random initialization, two updates.

Not a formal trainer or performance experiment. Run through scripts/with_gpu_lock.py.
"""
import argparse
import json
from pathlib import Path
import random
import subprocess
import time

import numpy as np
from PIL import Image
import torch

from .data import load_training_package, map_record
from .model import build_model, check_cuda_extension
from .predict import export_package, predict_instances, sha256, write_json


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--config', default='configs/runs/segmentation/pilot_smoke.json')
    parser.add_argument('--data-only', action='store_true')
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    records, report = load_training_package(args.package)
    mapped = [map_record(r, config['short_edge'], config['max_size']) for r in records]
    report['mapped_instances'] = sum(len(r['instances']) for r in mapped)
    report['mapped_images'] = len(mapped)
    write_json(output / 'data_check.json', report)
    if args.data_only:
        print(json.dumps(report))
        return
    if config['optimizer_updates'] != 2 or config['initialization'] != 'random_no_pretrained_weights':
        raise ValueError('This entry point only permits the two-update random-init check')
    seed = config['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    started = time.monotonic()
    model, cfg = build_model(args.source)
    kernel_check = check_cuda_extension()
    write_json(output / 'cuda_extension_check.json', kernel_check)
    (output / 'model_config.yaml').write_text(cfg.dump())
    write_json(output / 'resolved_config.json', config)
    groups = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            groups.append(dict(params=[param], lr=config['learning_rate'] *
                          (config['backbone_lr_multiplier'] if name.startswith('backbone.') else 1)))
    optimizer = torch.optim.AdamW(groups, weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    scaler = torch.cuda.amp.GradScaler(enabled=config['amp'])
    torch.cuda.reset_peak_memory_stats()
    losses_logged = []
    for update in range(2):
        if time.monotonic() - started > config['budget_seconds']:
            raise TimeoutError('Smoke time budget exceeded')
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=config['amp']):
            losses = model([mapped[update]])
            loss = sum(losses.values())
        if not torch.isfinite(loss):
            raise ValueError('Non-finite loss')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip_norm'], error_if_nonfinite=True)
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < scale_before:
            raise ValueError('AMP overflow skipped the optimizer update')
        scheduler.step()
        losses_logged.append(dict(update=update+1, image_id=records[update]['image_id'],
                                  total_loss=float(loss), gradient_norm=float(norm),
                                  components={k: float(v) for k, v in losses.items()}))
        if update == 0:
            checkpoint = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                              scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                              rng=rng_state(), update=1)
            torch.save(checkpoint, output / 'resume.pth')
            expected = model.sem_seg_head.predictor.class_embed.weight.detach().clone()
            # Build a fresh model/optimizer/scaler so loading is not a no-op.
            del checkpoint, optimizer, scheduler, scaler, model, groups, loss, losses
            restored, _ = build_model(args.source)
            model = restored
            groups = [dict(params=[p], lr=config['learning_rate'] *
                      (config['backbone_lr_multiplier'] if n.startswith('backbone.') else 1))
                      for n, p in model.named_parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(groups, weight_decay=config['weight_decay'])
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
            scaler = torch.cuda.amp.GradScaler(enabled=config['amp'])
            saved = torch.load(output / 'resume.pth', map_location='cpu')
            model.load_state_dict(saved['model'], strict=True)
            optimizer.load_state_dict(saved['optimizer'])
            scheduler.load_state_dict(saved['scheduler'])
            scaler.load_state_dict(saved['scaler'])
            restore_rng(saved['rng'])
            if saved['update'] != 1 or not torch.equal(model.sem_seg_head.predictor.class_embed.weight, expected):
                raise ValueError('Resume verification failed')
            if scheduler.state_dict() != saved['scheduler'] or scaler.state_dict() != saved['scaler']:
                raise ValueError('Scheduler/scaler resume mismatch')
            del saved
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(), scaler=scaler.state_dict(), rng=rng_state(), update=2),
               output / 'smoke_final.pth')
    model.eval()
    image = np.asarray(Image.open(records[0]['file_name']).convert('RGB'))
    inference_config = {k: config[k] for k in ('short_edge', 'max_size', 'score_threshold', 'max_candidates')}
    inference_config['image_id'] = records[0]['image_id']
    candidates = predict_instances(image, model, inference_config)
    # A new subset manifest retains the real input identity and provenance.
    manifest = json.loads((Path(args.package) / 'image_manifest.json').read_text(encoding='utf-8'))
    manifest['manifest_id'] += '-segmentation-smoke-one-image'
    manifest['images'] = [i for i in manifest['images'] if i['image_id'] == records[0]['image_id']]
    subset = output / 'input'
    subset.mkdir()
    import shutil
    for im in manifest['images']:
        dest = subset / im['file_name']
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(args.package) / im['file_name'], dest)
    write_json(subset / 'image_manifest.json', manifest)
    export_package(output / 'candidate_package', subset / 'image_manifest.json',
        {records[0]['image_id']: candidates}, candidate_set_id=output.name+'-random-smoke',
        producer_run_id=output.name, checkpoint_sha256=sha256(output / 'smoke_final.pth'),
        decoder_config_sha256=sha256(output / 'resolved_config.json'), illustrative_only=True,
        note='Real image, random-init model after TWO engineering updates; not usable S0 predictions or performance evidence.')
    torch.cuda.synchronize()
    result = dict(purpose=config['purpose'], status='completed', data=report,
        updates=losses_logged, resume_from_update=1, completed_updates=2,
        candidate_count=len(candidates), candidate_package_illustrative_only=True,
        elapsed_seconds=time.monotonic()-started,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
        checkpoint_sha256=sha256(output / 'smoke_final.pth'),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        pretrained_weights_loaded=False, performance_evaluated=False, cuda_extension_check=kernel_check)
    write_json(output / 'smoke_report.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
