"""Actual dev250 offline export/consumer receipt. No torch, model or GT inputs.

Run with CUDA_VISIBLE_DEVICES='' under timeout --kill-after=5s 1800s.
Arguments: RUN_DIRECTORY SEGMENTATION_CHECKOUT. Inputs/outputs stay on server.
"""
from collections import Counter
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from fashion_grounding.data.schema import read_json, write_json, sha256, require


def main():
    run, checkout = map(Path, sys.argv[1:])
    started = time.monotonic()
    report = dict(status='running', started_at=datetime.now(timezone.utc).isoformat(),
                  budget_seconds=1800, model_inference_calls=0, checkpoint_loads=0,
                  optimizer_updates=0, selected_threshold=None, packages=[])
    write_json(run/'offline_receipt.json', report)
    try:
        cache = read_json(run/'cache/cache.json')
        cache_hash = sha256(run/'cache/cache.json')
        require(cache_hash == (run/'cache/cache.sha256').read_text().strip(), 'Cache hash mismatch')
        require(cache['status'] == 'complete' and cache['illustrative_only'] is False, 'Real complete cache required')
        consumer = run/'grounding_inputs.py'
        require(sha256(consumer) == '74153d2e43d34379d9f6b9b0c9537742402010b00c140935ada7191d03cb13a8',
                'Consumer differs from pinned wyz 103dcc5')
        spec = importlib.util.spec_from_file_location('pinned_grounding_inputs', consumer)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        records = {}
        for ref in cache['records']:
            path = run/'cache'/ref['path']
            require(sha256(path) == ref['sha256'], 'Record hash mismatch')
            records[ref['image_id']] = read_json(path)
        require(len(records) == 250, 'Expected complete dev250')
        report.update(source_cache_sha256=cache_hash,
            source_image_manifest_sha256=cache['provenance']['source_image_manifest_sha256'],
            checkpoint_sha256=cache['provenance']['checkpoint_sha256'],
            cache_images=len(records),
            cache_candidates=sum(len(r['candidates']) for r in records.values()),
            cache_discarded_empty_masks=sum(r['discarded_empty_masks'] for r in records.values()),
            consumer_commit='103dcc546714dd7c7866b4259d96a9825d8090ef', consumer_sha256=sha256(consumer),
            script_sha256=sha256(__file__))
        previous_ids = None
        for threshold in (.3, .5, .7):
            output = run/f'trial-{threshold}'
            command = [sys.executable, '-m', 'fashion_grounding.segmentation.export_candidates', 'export',
                '--config', str(checkout/'configs/runs/segmentation/candidate_u3000_export.json'),
                '--cache', str(run/'cache'), '--output', str(output), '--threshold', str(threshold),
                '--candidate-set-id', f'u3000-dev250-20261114-r01-trial-{threshold}']
            stage_start = time.monotonic()
            with (run/f'export-{threshold}.log').open('x') as stream:
                subprocess.run(command, cwd=checkout, stdout=stream, stderr=subprocess.STDOUT,
                               check=True, timeout=max(1, 1750-(time.monotonic()-started)))
            pack = module.CandidatePack(output/'candidates_predicted.json')
            require(set(pack.images) == set(pack.rows) == set(records), 'Missing/extra image rows')
            counts, categories, ids = [], Counter(), set()
            crop_count = 0
            for iid in sorted(records):
                require(time.monotonic()-started < 1750, 'Consumer time budget exceeded')
                expected = [c for c in records[iid]['candidates'] if c['score'] >= float(np.float32(threshold))]
                require(pack.rows[iid]['candidates'] == expected, 'Filtering changed query identity or payload')
                image, candidates = pack.load(iid)
                crops = module.crop_regions(image, candidates)
                require(len(crops) == len(candidates) and all(min(c.size) > 0 for c in crops), 'Invalid crop')
                require([c.decoder_query_id for c in candidates] == [c['decoder_query_id'] for c in expected],
                        'Consumer query IDs changed')
                counts.append(len(candidates)); crop_count += len(crops)
                categories.update(c.category_id for c in candidates)
                ids.update(c.candidate_id for c in candidates)
                del image, candidates, crops
            require(len(ids) == sum(counts), 'Candidate ID collision')
            require(previous_ids is None or ids <= previous_ids, 'Threshold sets are not nested')
            previous_ids = ids
            report['packages'].append(dict(threshold=threshold, threshold_status='trial_unselected',
                path=str(output), command=command, images=len(counts), candidates=sum(counts),
                no_candidates_images=counts.count(0), validated_crops=crop_count,
                candidates_per_image=dict(min=min(counts), max=max(counts), mean=float(np.mean(counts)),
                                          median=float(np.median(counts))),
                candidate_counts_by_category={str(k):v for k,v in sorted(categories.items())},
                candidate_manifest_sha256=sha256(output/'candidates_predicted.json'),
                image_manifest_sha256=sha256(output/'image_manifest.json'),
                decoder_config_file_sha256=sha256(output/'resolved_decoder_config.json'),
                export_metadata_sha256=sha256(output/'run_metadata.json'),
                elapsed_export_and_consumer_seconds=time.monotonic()-stage_start,
                status='passed', query_ids_and_nested_threshold_sets_verified=True))
            write_json(run/'offline_receipt.json', report)
            print(json.dumps(report['packages'][-1]), flush=True)
        require('torch' not in sys.modules, 'Offline consumer unexpectedly imported torch')
        report.update(status='completed', elapsed_seconds=time.monotonic()-started,
                      finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        report.update(status='failed', error_code=type(error).__name__, elapsed_seconds=time.monotonic()-started)
        raise
    finally:
        write_json(run/'offline_receipt.json', report)


if __name__ == '__main__':
    main()
