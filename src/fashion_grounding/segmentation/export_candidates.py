"""Pinned selected-checkpoint candidates: GPU cache and offline CPU filtering.

Run cache under scripts/with_gpu_lock.py with a hard timeout. Merely importing
this module, --help and export never import PyTorch or construct/load a model.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image

from ..data.schema import read_json, write_json, sha256, safe_path, require
from .candidate_cache import CacheWriter, export_cached, image_index, fingerprint


def validate_config(config):
    updates = config['completed_updates']
    require(type(updates) is int and updates > 0 and
            (config['purpose'] == 'selected_checkpoint_candidate_export' or
             (config['purpose'] == 'u3000_candidate_export' and updates == 3000)),
            'Expected a pinned selected-checkpoint export config')
    decoder = config['decoder']
    require(decoder == dict(short_edge=800, max_size=1333, amp=False,
        decoder='one_foreground_class_per_query', mask_logit_threshold=0,
        score='foreground_probability_times_positive_mask_sigmoid_mean',
        empty_mask_policy='discard', score_threshold=0, max_candidates=100, mask_nms=False),
        'Unsupported decoder configuration; review before changing cache semantics')
    require(config['threshold_options'] == [0.3, 0.5, 0.7] and config['selected_threshold'] is None,
            'This entry produces threshold trials, not accepted-dev selection')
    if 'score_components' in config:
        require(config['score_components'] == ['class_probability', 'mask_quality'], 'Unknown score components')
    return config


def bind_selected_config(template, selection, report):
    """Resolve the post-training export by checkpoint hash, not step number."""
    from .evaluation import choose_checkpoint
    best = choose_checkpoint([report])
    chosen = selection['selected_checkpoint']
    require(selection['stop_reason'] == 'max_updates' and
            all(chosen[k] == best[k] for k in ('checkpoint_sha256', 'checkpoint_path',
                                                'completed_updates', 'dev_mask_AP')),
            'Report is not the final selected checkpoint')
    dev = template['inputs']['dev']
    require(report['evaluated_images'] == dev['images']
            and report['evaluation_data']['image_manifest_sha256'] == dev['image_manifest_sha256']
            and report['training_data'] == template['training_data'], 'Wrong export population')
    return dict(template, checkpoint_sha256=chosen['checkpoint_sha256'],
                completed_updates=chosen['completed_updates'], training_git_commit=report['training_git_commit'],
                training_config_sha256=report['training_config_sha256'])


def input_manifest(package, config, split):
    """No annotation or query file is opened, including during preflight."""
    path = Path(package)/'image_manifest.json'
    pin = config['inputs'][split]
    require(sha256(path) == pin['image_manifest_sha256'], 'Input manifest hash mismatch')
    manifest = read_json(path)
    images = image_index(manifest)
    require(manifest['illustrative_only'] is False and len(images) == pin['images'] and
            all(im['split'] == split for im in images.values()), 'Wrong real train/dev input')
    return path, images


def validate_checkpoint_metadata(state, config):
    require(state['completed_updates'] == config['completed_updates'] and
            state['scheduler']['completed_updates'] == config['completed_updates'], 'Wrong checkpoint update')
    require(state['config']['purpose'] == 's0_training' and
            state['config_sha256'] == config['training_config_sha256'] == fingerprint(state['config']),
            'Wrong training config')
    require(state['git_commit'] == config['training_git_commit'], 'Wrong training source')
    require(state['data'] == config['training_data'], 'Wrong checkpoint training data')


def cache_images(writer, predictor, config, *, deadline):
    """Streaming orchestration; injected predictor permits model-free unit tests."""
    for iid, im in sorted(writer.images.items()):
        require(time.monotonic() < deadline, 'Candidate cache time budget exceeded')
        path = safe_path(writer.source.parent, im['file_name'])
        require(sha256(path) == im['sha256'], 'Image hash mismatch')
        with Image.open(path) as image:
            require(image.size == (im['width'], im['height']), 'Image size mismatch')
            pixels = np.asarray(image.convert('RGB'))
        diagnostics = {}
        try:
            candidates = predictor(pixels, dict(config['decoder'], image_id=iid), diagnostics)
            writer.append(iid, candidates, diagnostics)
        except Exception as error:
            write_json(writer.root/'run_metadata.json', dict(status='failed', image_id=iid,
                completed_images=len(writer.records), error_code=type(error).__name__,
                message='Candidate inference/cache failed; partial output is ineligible'))
            raise
        # Release full-resolution masks before processing the next image.
        del candidates, pixels
        require(time.monotonic() < deadline, 'Candidate cache time budget exceeded')
    return writer.finish()


def generate_cache(args, config):
    require(args.budget_seconds > 0, 'Positive time budget required')
    deadline = time.monotonic()+args.budget_seconds
    manifest, _ = input_manifest(args.package, config, args.split)
    require(sha256(Path(__file__).with_name('predict.py')) == config['source_decoder_sha256'],
            'Decoder changed; review and pin a new cache version')
    checkpoint = Path(args.checkpoint)
    require(sha256(checkpoint) == config['checkpoint_sha256'], 'Checkpoint hash mismatch')
    require(not Path(args.output).exists(), 'Output already exists')
    repo = Path(__file__).resolve().parents[3]
    source_commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    require(not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain',
            '--untracked-files=normal'], text=True).strip(), 'Commit export source before inference')
    # Heavy imports only occur in this explicitly requested GPU subcommand.
    import torch
    from .model import build_model
    from .predict import predict_instances
    from .training_state import state_digest
    torch.set_num_threads(2)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    validate_checkpoint_metadata(state, config)
    require(state_digest(state['model']) == state['component_digests']['model'], 'Model fingerprint mismatch')
    model, model_config = build_model(args.source, device='cpu')
    model.load_state_dict(state['model'], strict=True)
    del state
    require(time.monotonic() < deadline, 'Candidate cache time budget exceeded during model loading')
    model.to('cuda').eval()
    writer = CacheWriter(args.output, manifest, dict(producer_run_id=args.run_id,
        checkpoint_sha256=config['checkpoint_sha256'], completed_updates=config['completed_updates'],
        training_git_commit=config['training_git_commit'], export_git_commit=source_commit,
        export_config_sha256=fingerprint(config), source_decoder_sha256=config['source_decoder_sha256'],
        started_at=datetime.now(timezone.utc).isoformat(),
        **({'score_components': config['score_components']} if 'score_components' in config else {})), config['decoder'])
    write_json(writer.root/'resolved_export_config.json', config)
    (writer.root/'model_config.yaml').write_text(model_config.dump())
    with torch.inference_mode():
        cache_images(writer, lambda pixels, settings, diag: predict_instances(
            pixels, model, settings, diagnostics=diag), config, deadline=deadline)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    bind = sub.add_parser('bind-selected', help='Prepare an export config from final selection and its dev report')
    for key in ('config', 'selection', 'report', 'output'):
        bind.add_argument('--'+key, required=True)
    cache = sub.add_parser('cache', help='GPU forward pass; use existing shared-GPU lock/timeout wrapper')
    for key in ('config', 'package', 'checkpoint', 'source', 'output', 'run-id'):
        cache.add_argument('--'+key, required=True)
    cache.add_argument('--split', choices=('train', 'dev'), required=True)
    cache.add_argument('--budget-seconds', type=int, required=True)
    export = sub.add_parser('export', help='Filter a completed cache without loading any model')
    for key in ('config', 'cache', 'output', 'candidate-set-id'):
        export.add_argument('--'+key, required=True)
    export.add_argument('--threshold', type=float, choices=(0.3, 0.5, 0.7), required=True)
    args = parser.parse_args()
    config = validate_config(read_json(args.config))
    if args.command == 'bind-selected':
        import json
        resolved = bind_selected_config(config, read_json(args.selection), read_json(args.report))
        with Path(args.output).open('x') as stream:
            json.dump(resolved, stream, indent=2, allow_nan=False)
            stream.write('\n')
    elif args.command == 'cache':
        generate_cache(args, config)
    else:
        # Reject different decoder/input/producer configuration before exporting.
        metadata = read_json(Path(args.cache)/'cache.json')
        require(metadata['provenance']['export_config_sha256'] == fingerprint(config) and
                metadata['provenance']['decoder_config_sha256'] == fingerprint(config['decoder']),
                'Wrong export config/cache version')
        require(metadata['provenance']['source_image_manifest_sha256'] in
                {v['image_manifest_sha256'] for v in config['inputs'].values()}, 'Unregistered cache input')
        export_cached(args.cache, args.output, threshold=args.threshold,
            candidate_set_id=args.candidate_set_id, expected_checkpoint_sha256=config['checkpoint_sha256'])


if __name__ == '__main__':
    main()
