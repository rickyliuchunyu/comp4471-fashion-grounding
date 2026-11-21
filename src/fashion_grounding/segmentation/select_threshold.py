"""CPU-only coverage/selection with explicit accepted or user-authorized dev basis.

No model, pixels, checkpoint, feature encoding, or prediction generation is used.
The shared wkq matcher is imported unchanged; association is evaluation-only.
"""
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import math
from pathlib import Path
import platform
import subprocess
import sys
import time

from ..data.schema import (decode_mask, read_json, require, safe_path, sha256,
                           unique_index, validate_queries, write_json)
from ..evaluation import matching
from .candidate_cache import digest, image_index


THRESHOLDS = (0.3, 0.5, 0.7)


def referenced(ref, base):
    """Explicit input references may be absolute or relative to their JSON file."""
    require(isinstance(ref, dict) and isinstance(ref.get('path'), str) and ref['path'],
            'Missing input reference')
    path = Path(ref['path'])
    if not path.is_absolute():
        path = Path(base)/path
    require(sha256(path) == digest(ref['sha256']), 'Input hash mismatch: '+str(path))
    return path.resolve()


def load_frozen_dev(path, config, *, allow_user_authorized_dev=False):
    """Validate new query/freeze metadata BEFORE opening any candidate masks."""
    path = Path(path).resolve()
    freeze = read_json(path)
    require(freeze.get('schema_version') == 1 and freeze.get('split') == 'dev',
            'Dev-only freeze receipt required')
    user_authorized = freeze.get('purpose') == 'user_authorized_language_dev'
    if user_authorized:
        require(allow_user_authorized_dev and freeze.get('development_freeze') is True and
                freeze.get('formal_freeze') is False,
                'Explicit user-authorized dev opt-in required; not an accepted release')
        authorization = read_json(referenced(freeze['authorization_record'], path.parent))
        require(authorization.get('schema_version') == 1 and
                authorization.get('authority') == 'current_user' and
                authorization.get('authorized_use') == 'offline_candidate_threshold_selection' and
                isinstance(authorization.get('user_statement'), str) and
                authorization['user_statement'].strip() and
                authorization.get('confirmed_by') == freeze.get('confirmed_by') and
                authorization.get('confirmed_at') == freeze.get('confirmed_at') and
                authorization.get('queries_sha256') == freeze['queries']['sha256'],
                'User authorization does not bind this dev input/use')
    else:
        require(freeze.get('purpose') == 'accepted_language_dev_freeze' and
                freeze.get('formal_freeze') is True,
                'Accepted language dev is not frozen')
    require(freeze.get('confirmed_by') and freeze.get('confirmed_at'),
            'Missing human freeze confirmation')
    if not user_authorized:
        evidence = freeze.get('review_records')
        require(isinstance(evidence, list) and evidence, 'Missing review records')
        for ref in evidence:
            require(referenced(ref, path.parent).stat().st_size > 0, 'Empty review record')
    refs = {key: referenced(freeze[key], path.parent)
            for key in ('queries', 'image_manifest', 'annotations')}
    require(sha256(refs['image_manifest']) == config['image_manifest_sha256'] and
            sha256(refs['annotations']) == config['annotation_sha256'],
            'Frozen dev differs from pinned segmentation input')
    manifest, coco = read_json(refs['image_manifest']), read_json(refs['annotations'])
    for obj in (manifest, coco):
        require(obj['schema_version'] == 3 and obj['illustrative_only'] is False,
                'Real schema-3 dev required')
    images = image_index(manifest)
    require(all(im['split'] == 'dev' for im in images.values()), 'Dev-only input required')
    coco_images = unique_index(coco['images'], 'id')
    require(set(coco_images) == set(images), 'COCO/manifest image mismatch')
    for iid, im in images.items():
        require(all(coco_images[iid][k] == im[k] for k in ('width', 'height')),
                'COCO image dimensions mismatch')
    annotations = unique_index(coco['annotations'], 'id')
    for aid, ann in annotations.items():
        require(type(aid) is int and aid > 0 and ann['image_id'] in images and
                type(ann['category_id']) is int and 1 <= ann['category_id'] <= 8,
                'Invalid GT identity/category')
    import json
    queries = [json.loads(line) for line in refs['queries'].read_text().splitlines() if line.strip()]
    require(queries, 'Empty dev: coverage and selected threshold remain null')
    require(all(isinstance(q.get(key), str) and q[key].strip()
                for q in queries for key in ('query_id', 'semantic_query_id')),
            'Nonempty string query identities required')
    validate_queries(queries, images, annotations, accepted_only=not user_authorized)
    require(all(q['split'] == 'dev' for q in queries), 'Queries must all be dev')
    ids = sorted({q['image_id'] for q in queries})
    scope = freeze['scope']
    subset_kind = 'user_authorized_dev_subset' if user_authorized else 'accepted_dev_subset'
    require(scope['kind'] in ('full_dev', subset_kind) and scope['description'],
            'Explicit evaluation scope required')
    require(type(scope['query_count']) is int and scope['query_count'] == len(queries) and
            scope['query_image_ids'] == ids, 'Frozen query denominator mismatch')
    if scope['kind'] == 'full_dev':
        require(set(ids) == set(images), 'Subset cannot be reported as full dev')
    return freeze, images, annotations, queries


def load_trials(config, base, images):
    specs = config['packages']
    require(len(specs) == 3 and sorted(p['threshold'] for p in specs) == list(THRESHOLDS),
            'Exactly the three registered threshold packages are required')
    trials = []
    for spec in sorted(specs, key=lambda p: p['threshold']):
        path = referenced(spec, base)
        package = read_json(path)
        require(package['schema_version'] == 3 and package['illustrative_only'] is False and
                package['candidate_source'] == 'predicted', 'Real predicted candidates required')
        p = package['provenance']
        require(p['checkpoint_sha256'] == config['checkpoint_sha256'] and
                p['source_cache_sha256'] == config['source_cache_sha256'] and
                p['source_image_manifest_sha256'] == config['image_manifest_sha256'] and
                p['score_threshold'] == spec['threshold'], 'Candidate provenance mismatch')
        manifest_path = safe_path(path.parent, package['image_manifest']['path'])
        require(sha256(manifest_path) == package['image_manifest']['sha256'],
                'Candidate image manifest hash mismatch')
        exported = image_index(read_json(manifest_path))
        rows = unique_index(package['images'], 'image_id')
        require(set(rows) == set(exported) == set(images), 'Incomplete candidate image inventory')
        for iid, im in images.items():
            # Export canonicalizes file names; compare identity/pixels, not path strings.
            require(all(im[k] == exported[iid][k]
                        for k in ('width', 'height', 'sha256', 'split', 'group_id')),
                    'Candidate/source image identity mismatch')
        trials.append(dict(threshold=spec['threshold'], path=path, package=package, images=rows,
                           rows=[], association={}))
    return trials


def image_coverage(queries, annotations, candidate_row, candidate_masks, gt_masks):
    """Match against ALL non-crowd GT in the image, not just queried targets."""
    candidates = candidate_row['candidates']
    associations = matching.associate(candidates, annotations, candidate_masks, gt_masks)
    inverse = {aid: cid for cid, aid in associations.items()}
    anns = {a['id']: a for a in annotations}
    rows = []
    for q in queries:
        target = anns[q['target_instance_id']]
        overlap = max((matching.mask_iou(mask, gt_masks[target['id']])
                       for mask in candidate_masks.values()), default=0.)
        rows.append(dict(query_id=q['query_id'], image_id=q['image_id'], language=q['language'],
            target_instance_id=target['id'], category_id=target['category_id'],
            area_bin='small' if target['area'] < 32**2 else
                     'medium' if target['area'] < 96**2 else 'large',
            candidate_count=len(candidates), status=candidate_row['status'],
            covered=target['id'] in inverse, matched_candidate_id=inverse.get(target['id']),
            geometric_recalled=overlap >= .5, max_target_mask_iou=overlap))
    return rows, associations


def summarize(rows):
    require(rows, 'Empty query denominator')
    counts = {}
    for row in rows:
        iid, count = row['image_id'], row['candidate_count']
        require(counts.setdefault(iid, count) == count, 'Inconsistent image candidate count')
    covered = sum(r['covered'] for r in rows)
    recalled = sum(r['geometric_recalled'] for r in rows)
    return dict(queries=len(rows), images=len(counts), covered_queries=covered,
        matched_candidate_coverage=covered/len(rows), geometric_recalled_queries=recalled,
        geometric_recall_at_0_5=recalled/len(rows),
        candidate_count_on_query_images=sum(counts.values()),
        mean_candidates_on_query_images=sum(counts.values())/len(counts),
        zero_candidate_images=sum(n == 0 for n in counts.values()))


def choose_threshold(results):
    require(len(results) == 3 and sorted(r['threshold'] for r in results) == list(THRESHOLDS),
            'Three threshold results required')
    ids = [set(r['query_id'] for r in trial['rows']) for trial in results]
    require(ids[0] and all(s == ids[0] for s in ids), 'Different query denominators')
    for trial in results:
        require(len(trial['rows']) == len(ids[0]), 'Duplicate query results')
        trial['summary'] = summarize(trial['rows'])
    best = min(results, key=lambda r: (-r['summary']['covered_queries'],
                r['summary']['candidate_count_on_query_images'], -r['threshold']))
    max_covered = best['summary']['covered_queries']
    finalists = [r for r in results if r['summary']['covered_queries'] == max_covered]
    if len(finalists) == 1:
        reason = 'most_covered_queries'
    else:
        minimum = min(r['summary']['candidate_count_on_query_images'] for r in finalists)
        reason = 'fewest_candidates_on_unique_query_images' if sum(
            r['summary']['candidate_count_on_query_images'] == minimum for r in finalists) == 1 else 'higher_threshold'
    return best['threshold'], reason


def run(config_path, freeze_path, output, *, budget_seconds=300, allow_user_authorized_dev=False):
    require(math.isfinite(budget_seconds) and budget_seconds > 0, 'Positive finite time budget required')
    deadline = time.monotonic()+budget_seconds
    def check_time():
        require(time.monotonic() < deadline, 'Offline coverage time budget exceeded')
    config_path, freeze_path, output = Path(config_path).resolve(), Path(freeze_path).resolve(), Path(output)
    require(not output.exists(), 'Output already exists; use a new run directory')
    config = read_json(config_path)
    require(config['schema_version'] == 1 and config['purpose'] == 'offline_candidate_threshold_selection',
            'Wrong selection config')
    require(sha256(matching.__file__) == config['matcher_sha256'], 'Shared matcher version changed')
    freeze, images, annotations, queries = load_frozen_dev(freeze_path, config,
        allow_user_authorized_dev=allow_user_authorized_dev)
    evaluation_basis = ('user_authorized_development' if
        freeze['purpose'] == 'user_authorized_language_dev' else 'accepted_language_dev')
    check_time()
    trials = load_trials(config, config_path.parent, images)
    by_image, anns_by_image = defaultdict(list), defaultdict(list)
    for q in queries:
        by_image[q['image_id']].append(q)
    for a in annotations.values():
        if not a['iscrowd']:
            anns_by_image[a['image_id']].append(a)
    output.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[3]
    git_commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = bool(subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True).strip())
    metadata = dict(status='running', started_at_utc=datetime.now(timezone.utc).isoformat(),
        record_schema_version=1, run_id=output.name, experiment_id='S0', owner='segmentation',
        split='dev', candidate_source='predicted', git_commit=git_commit, source_worktree_dirty=dirty,
        command=sys.argv, query_manifest_sha256=freeze['queries']['sha256'],
        checkpoint_sha256=config['checkpoint_sha256'],
        config_sha256=sha256(config_path), freeze_sha256=sha256(freeze_path),
        source_sha256=sha256(__file__), matcher_sha256=sha256(matching.__file__),
        python=platform.python_version(), budget_seconds=budget_seconds,
        model_inference_calls=0, checkpoint_loads=0, optimizer_updates=0,
        evaluation_basis=evaluation_basis,
        source_review_status_counts={status: sum(q.get('review_status') == status for q in queries)
            for status in sorted({q.get('review_status', 'missing') for q in queries})})
    from ..data import schema
    metadata['schema_source_sha256'] = sha256(schema.__file__)
    write_json(output/'run_metadata.json', metadata)
    try:
        for iid in sorted(by_image):
            check_time()
            im = images[iid]
            aa = anns_by_image[iid]
            gt_masks = {a['id']: decode_mask(a['segmentation'], im['height'], im['width']) for a in aa}
            # Three nested packages share masks. Decode each hash at most once per image.
            decoded = {}
            for trial in trials:
                row = trial['images'][iid]
                require(row['image_size_wh'] == [im['width'], im['height']], 'Candidate dimensions mismatch')
                cs = row['candidates']
                unique_index(cs, 'candidate_id')
                require(row['status'] == ('ok' if cs else 'no_candidates') or
                        (row['status'] == 'inference_error' and not cs), 'Invalid candidate status')
                masks = {}
                for c in cs:
                    key = c['mask_sha256']
                    if key not in decoded:
                        mask_path = safe_path(trial['path'].parent, c['mask_reference'])
                        require(sha256(mask_path) == key, 'Candidate mask hash mismatch')
                        decoded[key] = decode_mask(read_json(mask_path), im['height'], im['width'])
                    masks[c['candidate_id']] = decoded[key]
                rr, assoc = image_coverage(by_image[iid], aa, row, masks, gt_masks)
                trial['rows'].extend(rr)
                trial['association'][str(iid)] = assoc
            # Release full-resolution arrays before advancing to the next image.
            del gt_masks, decoded, masks
        check_time()
        selected, reason = choose_threshold(trials)
        results = []
        for trial in trials:
            groups = {}
            for field in ('language', 'category_id', 'area_bin'):
                buckets = defaultdict(list)
                for row in trial['rows']:
                    buckets[str(row[field])].append(row)
                groups[field] = {key: summarize(rows) for key, rows in sorted(buckets.items())}
            result = dict(threshold=trial['threshold'], evaluation_basis=evaluation_basis,
                          summary=trial['summary'], groups=groups,
                          rows=trial['rows'], association=trial['association'])
            write_json(output/f'coverage-{trial["threshold"]}.json', result)
            results.append(dict(threshold=trial['threshold'], **trial['summary']))
        chosen = next(t for t in trials if t['threshold'] == selected)
        selection = dict(status=('selected_for_user_authorized_development' if
            evaluation_basis == 'user_authorized_development' else 'selected'),
            evaluation_basis=evaluation_basis,
            source_review_status_counts=metadata['source_review_status_counts'],
            selected_threshold=selected, selection_reason=reason,
            scope=freeze['scope'], freeze_sha256=metadata['freeze_sha256'],
            queries_sha256=freeze['queries']['sha256'], image_manifest_sha256=config['image_manifest_sha256'],
            annotation_sha256=config['annotation_sha256'], checkpoint_sha256=config['checkpoint_sha256'],
            source_cache_sha256=config['source_cache_sha256'], matcher_sha256=metadata['matcher_sha256'],
            selected_package=dict(path=str(chosen['path']), sha256=sha256(chosen['path']),
                                  candidate_set_id=chosen['package']['candidate_set_id']),
            comparison=results, config_sha256=metadata['config_sha256'],
            note='Evaluation association only; immutable input trial packages are not rewritten.')
        with (output/'comparison.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
        write_json(output/'resolved_config.json', config)
        write_json(output/'resolved_freeze.json', freeze)
        check_time()
        # Only a complete run gets a consumable selection record.
        write_json(output/'selection.json', selection)
        metadata.update(status='completed', selected_threshold=selected,
                        elapsed_seconds=budget_seconds-(deadline-time.monotonic()),
                        finished_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(output/'run_metadata.json', metadata)
        return selection
    except Exception as exc:
        (output/'selection.json').unlink(missing_ok=True)
        metadata.update(status='failed', selected_threshold=None, error=str(exc))
        write_json(output/'run_metadata.json', metadata)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'freeze', 'output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--budget-seconds', type=float, default=300)
    parser.add_argument('--user-authorized-dev', action='store_true',
        help='Use an explicitly authorized fixed dev receipt; preserve upstream review status')
    args = parser.parse_args()
    result = run(args.config, args.freeze, args.output, budget_seconds=args.budget_seconds,
                 allow_user_authorized_dev=args.user_authorized_dev)
    print('Selected threshold:', result['selected_threshold'], result['selection_reason'])


if __name__ == '__main__':
    main()
