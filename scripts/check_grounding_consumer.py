"""Read-only acceptance against wyz's exact published reader, without merging it."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from fashion_grounding.data.schema import read_json, sha256, write_json, require

PINNED_GROUNDING_COMMIT = '66c0da7e066c1235ee618ea7015171e73b82c914'
READER_PATH = 'src/fashion_grounding/grounding/inputs.py'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grounding-repo', required=True)
    parser.add_argument('--candidates', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source = subprocess.check_output(['git', '-C', args.grounding_repo, 'show', PINNED_GROUNDING_COMMIT+':'+READER_PATH])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    reports = []
    with tempfile.TemporaryDirectory() as tmp:
        module_path = Path(tmp)/'published_inputs.py'
        module_path.write_bytes(source)
        spec = importlib.util.spec_from_file_location('published_grounding_inputs', module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        for path in args.candidates:
            pack = module.CandidatePack(path, allow_illustrative=True)
            expected = read_json(path)
            default_rejected = False
            if expected['illustrative_only']:
                try:
                    module.CandidatePack(path)
                except ValueError:
                    default_rejected = True
                require(default_rejected, 'Diagnostic package accepted without explicit opt-in')
            count, statuses = 0, {}
            for iid, row in pack.rows.items():
                image, candidates = pack.load(iid)
                crops = module.crop_regions(image, candidates)
                require([c.candidate_id for c in candidates] == [c['candidate_id'] for c in row['candidates']], 'Order changed')
                for crop, candidate in zip(crops, candidates):
                    x1, y1, x2, y2 = candidate.bbox_xyxy
                    require(crop.size == (x2-x1, y2-y1), 'Crop geometry mismatch')
                    require(not hasattr(candidate, 'source_annotation_id'), 'GT reference leaked to inference')
                count += len(candidates)
                statuses[row['status']] = statuses.get(row['status'], 0)+1
            reports.append(dict(candidate_manifest_sha256=sha256(path), path=str(Path(path).resolve()),
                images=len(pack.rows), candidates=count, statuses=statuses, default_diagnostic_rejection=default_rejected,
                image_mask_hash_geometry_crop_and_order_checks='passed'))
    report = dict(status='passed', grounding_commit=PINNED_GROUNDING_COMMIT, reader_path=READER_PATH,
                  reader_sha256=hashlib.sha256(source).hexdigest(), packages=reports,
                  scope='candidate consumption only; no encoder, language accuracy or dev acceptance')
    write_json(output/'consumer_report.json', report)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
