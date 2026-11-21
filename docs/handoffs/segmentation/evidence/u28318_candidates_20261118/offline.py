"""Export and verify three CPU trials from an already complete dev cache."""
import argparse
import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
from fashion_grounding.data.schema import read_json, require, sha256, write_json
from fashion_grounding.segmentation.select_threshold import run as select_threshold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'export-config', 'coverage-template', 'freeze', 'reader'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--release-id', required=True)
    parser.add_argument('--reader-sha256', required=True)
    args = parser.parse_args()
    root = args.run.resolve()
    cache_dir = root / 'cache'
    cache = read_json(cache_dir / 'cache.json')
    cache_digest = sha256(cache_dir / 'cache.json')
    config = read_json(args.export_config)
    require(cache['status'] == 'complete' and len(cache['records']) == config['inputs']['dev']['images'],
            'Expected the complete dev cache')
    require(cache['provenance']['checkpoint_sha256'] == config['checkpoint_sha256'], 'Wrong checkpoint')
    require(sha256(args.reader) == args.reader_sha256, 'Wrong downstream reader revision')
    spec = importlib.util.spec_from_file_location('release_candidate_reader', args.reader)
    consumer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = consumer
    spec.loader.exec_module(consumer)
    cache_rows = {ref['image_id']: read_json(cache_dir / ref['path']) for ref in cache['records']}
    coverage = read_json(args.coverage_template)
    coverage.update(source_cache_sha256=cache_digest, checkpoint_sha256=config['checkpoint_sha256'], packages=[])
    receipt = dict(checkpoint_sha256=config['checkpoint_sha256'], source_cache_sha256=cache_digest,
                   reader_sha256=args.reader_sha256, status='complete', images=len(cache_rows),
                   cached_candidates=sum(len(row['candidates']) for row in cache_rows.values()), trials=[])
    for cutoff in config['threshold_options']:
        directory = root / f'trial-{cutoff}'
        manifest = directory / 'candidates_predicted.json'
        subprocess.run([sys.executable, '-m', 'fashion_grounding.segmentation.export_candidates', 'export',
                        '--config', str(args.export_config), '--cache', str(cache_dir), '--output', str(directory),
                        '--threshold', str(cutoff), '--candidate-set-id', f'{args.release_id}-trial-{cutoff}'],
                       check=True)
        pack = consumer.CandidatePack(manifest)
        require(pack.rows.keys() == pack.images.keys() == cache_rows.keys(), 'Incomplete dev inventory')
        total = empty = 0
        for image_id, row in pack.rows.items():
            expected = [c for c in cache_rows[image_id]['candidates'] if c['score'] >= float(np.float32(cutoff))]
            require(row['candidates'] == expected, 'Cache projection changed query ID, score, box or mask')
            image, candidates = pack.load(image_id)
            crops = consumer.crop_regions(image, candidates)
            require(len(crops) == len(expected) and all(min(c.size) > 0 for c in crops), 'Invalid crop')
            total += len(crops)
            empty += not crops
        entry = dict(threshold=cutoff, path=str(manifest), sha256=sha256(manifest))
        coverage['packages'].append(entry)
        receipt['trials'].append(dict(entry, images=len(pack.rows), candidates=total, zero_candidate_images=empty))
        print(f'Validated {cutoff}: {total} candidates on {len(pack.rows)} images', flush=True)
    require('torch' not in sys.modules, 'Offline validation unexpectedly loaded torch')
    write_json(root / 'coverage_config.json', coverage)
    select_threshold(root / 'coverage_config.json', args.freeze, root / 'selection', budget_seconds=180,
                     allow_user_authorized_dev=True)
    write_json(root / 'offline_receipt.json', receipt)


if __name__ == '__main__':
    main()
