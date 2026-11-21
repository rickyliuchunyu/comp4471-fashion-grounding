"""Bounded lcy adapter around wyz's unchanged published B1 interface."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--user-authorized-dev', action='store_true')
    args = p.parse_args()
    config = json.loads(Path(args.config).read_text())
    if not args.user_authorized_dev:
        p.error('This dev run requires explicit user authorization')
    repo = Path(__file__).resolve().parents[1]
    grounding = Path(config['grounding_repo'])
    head = subprocess.check_output(['git', '-C', str(grounding), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(grounding), 'status', '--porcelain'], text=True).strip()
    if head != config['grounding_commit'] or dirty:
        raise ValueError('Published grounding source changed; do not use teammate work in progress')
    # Dependency code stays in the teammate checkout, read-only. No merge or method edits.
    sys.path.insert(0, str(grounding/'src'))
    import fashion_grounding
    fashion_grounding.__path__.append(str(repo/'src/fashion_grounding'))
    from fashion_grounding.data.schema import read_json, write_json, sha256, require
    from fashion_grounding.grounding.inputs import CandidatePack
    from fashion_grounding.grounding.predict import read_queries, stage_inputs, execute
    from fashion_grounding.grounding.cache import FeatureCache
    from fashion_grounding.segmentation.pipeline import resolve_outputs, evaluate_saved_associations

    def fixed(name):
        ref = config[name]
        require(sha256(ref['path']) == ref['sha256'], 'Changed input: '+name)
        return Path(ref['path'])

    selection = read_json(fixed('selection'))
    require(selection['selected_threshold'] == .3 and
            selection['selected_package']['sha256'] == config['candidates']['sha256'] and
            selection['queries_sha256'] == config['queries']['sha256'] and
            selection['annotation_sha256'] == config['annotations']['sha256'], 'Wrong selected input')
    auth = config['authorization']
    require(auth['user_statement'] and auth['scope'] == 'lcy_minimal_pipeline_dev_evaluation' and
            auth['queries_sha256'] == config['queries']['sha256'], 'Wrong authorization scope')
    pack = CandidatePack(fixed('candidates'))
    require(pack.package['candidate_source'] == 'predicted', 'Predicted candidates only')
    queries_path = fixed('queries')
    queries = read_queries(queries_path, pack, draft=True)
    require(len(queries) == 290 and len({q.image_id for q in queries}) == 116 and
            all(q.split == 'dev' for q in queries), 'Wrong dev scope')
    # read_queries constructs wyz's answer-free Query allowlist. No GT loaded for ranking.
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    meta = dict(record_schema_version=1, run_id=output.name, experiment_id='B1',
        owner='segmentation', executed_by='lcy', role='integration_only', status='running',
        split='dev', candidate_source='predicted', evaluation_basis='user_authorized_development',
        started_at=datetime.now(timezone.utc).isoformat(), grounding_commit=head,
        git_commit=subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
        source_worktree_dirty=bool(subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True).strip()),
        command=[sys.executable, *sys.argv], config_sha256=sha256(args.config),
        candidate_manifest_sha256=pack.digest, query_manifest_sha256=sha256(queries_path),
        segmentation_inference_calls=0, optimizer_updates=0, old_smoke_runs=0)
    write_json(output/'run_metadata.json', meta)
    write_json(output/'resolved_config.json', config)
    try:
        ref = stage_inputs(pack, output)
        from fashion_grounding.encoders.siglip import SiglipEncoder
        raw_encoder = SiglipEncoder(config['snapshot'], config['revision'], 'cuda', 8)
        counts = Counter()

        class CountEncoder:
            def __getattr__(self, name): return getattr(raw_encoder, name)
            def encode_images(self, images):
                counts['encoded_regions'] += len(images)
                return raw_encoder.encode_images(images)
            def encode_texts(self, texts):
                counts['encoded_texts'] += len(texts)
                return raw_encoder.encode_texts(texts)

        class ReuseCache:
            def __init__(self):
                self.local = FeatureCache(output/'features')
                require(Path(config['existing_cache']).is_dir(), 'Existing cache missing')
                self.shared = FeatureCache(config['existing_cache'])
            def get(self, spec, rows, dimension):
                for label, cache in [('local', self.local), ('existing', self.shared)]:
                    value = cache.get(spec, rows, dimension)
                    if value is not None:
                        counts[label+'_'+spec['kind']+'_hits'] += 1
                        return value
                counts[spec['kind']+'_misses'] += 1
                return None
            def put(self, spec, value): self.local.put(spec, value)

        records = execute(pack, queries, CountEncoder(), ReuseCache(), output.name, 'B1', ref)
        predictions = [r for r, _ in records]
        errors = [e for _, e in records if e]
        def jsonl(name, values):
            with (output/name).open('x') as f:
                for v in values: f.write(json.dumps(v, ensure_ascii=False, allow_nan=False)+'\n')
        jsonl('predictions.jsonl', predictions)
        write_json(output/'errors.json', errors)
        outputs = resolve_outputs(predictions, pack.package, ref)
        jsonl('resolved_outputs.jsonl', outputs)
        # Verify the new geometry views reference the exact original bytes, never a refined mask.
        for out in outputs:
            if out['geometry']:
                g = out['geometry']
                require(sha256(output/g['mask_reference']) == g['mask_sha256'], 'Output mask changed')
        meta.update(encoder=raw_encoder.signature, cache_counts=dict(counts), prediction_seconds=time.monotonic()-started)
        del raw_encoder
        import torch
        torch.cuda.empty_cache()
        # GT and frozen associations enter only AFTER the prediction file is complete.
        coverage = read_json(fixed('coverage'))
        require(coverage['threshold'] == .3 and coverage['summary']['queries'] == 290 and
                coverage['summary']['covered_queries'] == 258, 'Wrong frozen coverage')
        coco = read_json(fixed('annotations'))
        raw_queries = [json.loads(line) for line in queries_path.read_text().splitlines() if line.strip()]
        report = evaluate_saved_associations(raw_queries, predictions, coverage, pack.package,
                                            pack.path, coco['annotations'])
        write_json(output/'evaluation.json', report)
        meta.update(status='completed' if not errors else 'failed', evaluation=report['summary'],
            query_count=len(predictions), source_review_status_counts=dict(Counter(q['review_status'] for q in raw_queries)),
            geometry_outputs=sum(o['geometry'] is not None for o in outputs), inference_errors=len(errors),
            artifacts={n: {'path': n, 'sha256': sha256(output/n)} for n in
                ('predictions.jsonl', 'resolved_outputs.jsonl', 'evaluation.json', 'errors.json', 'resolved_config.json')})
        require(sha256(queries_path) == config['queries']['sha256'], 'Source queries changed')
        require(not errors, 'Ranking errors retained in evaluation; inspect errors.json')
    except BaseException as exc:
        meta.update(status='failed', failure={'type': type(exc).__name__, 'message': str(exc)})
        raise
    finally:
        meta.update(finished_at=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-started)
        write_json(output/'run_metadata.json', meta)
    print(json.dumps({'status': meta['status'], 'summary': meta['evaluation'], 'seconds': meta['elapsed_seconds']}))


if __name__ == '__main__':
    main()
