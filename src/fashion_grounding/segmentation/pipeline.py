"""lcy integration only: resolve original geometry and score frozen associations.

No segmentation, ranking, association rebuilding, or review-label mutation.
"""
from collections import Counter
from pathlib import Path

from ..data.schema import decode_mask, read_json, require, safe_path, sha256, unique_index
from ..evaluation.matching import mask_iou


def resolve_outputs(predictions, package, manifest_ref):
    images = unique_index(package['images'], 'image_id')
    unique_index(predictions, 'query_id')
    outputs = []
    for p in predictions:
        require(p['candidate_source'] == 'predicted' and
                p['candidate_set_id'] == package['candidate_set_id'] and
                p['candidate_manifest'] == manifest_ref, 'Wrong candidate identity')
        row = images[p['image_id']]
        candidates = unique_index(row['candidates'], 'candidate_id')
        selected = p['selected_candidate']
        out = {k: p[k] for k in ('query_id', 'image_id', 'status', 'candidate_set_id')}
        out.update(candidate_manifest=manifest_ref, geometry=None)
        require(p['status'] in ('success', 'no_candidates', 'no_output', 'rule_unsupported',
                                'rejected', 'inference_error'), 'Unknown prediction status')
        if row['status'] != 'ok':
            require(p['status'] == row['status'], 'Upstream failure not propagated')
        if p['status'] == 'success':
            require(isinstance(selected, dict) and selected['candidate_id'] in candidates,
                    'Selected candidate not in this image')
            c = candidates[selected['candidate_id']]
            # A derived display view. The authoritative schema-3 prediction retains only ID/score.
            out['geometry'] = {k: c[k] for k in ('candidate_id', 'category_id', 'bbox_xyxy',
                'mask_reference', 'mask_sha256', 'decoder_query_id')}
            out['geometry']['mask_reference'] = str(Path(manifest_ref['path']).parent/c['mask_reference'])
        else:
            require(selected is None, 'Failure contains selected candidate')
            if p['status'] == 'no_candidates':
                require(row['status'] == 'no_candidates', 'False empty result')
        outputs.append(out)
    return outputs


def summarize(rows):
    require(rows, 'Empty denominator')
    n = len(rows)
    covered = sum(r['covered'] for r in rows)
    correct = sum(r['correct'] for r in rows)
    require(all(not r['correct'] or r['covered'] for r in rows), 'Correct outside coverage')
    return dict(queries=n, images=len({r['image_id'] for r in rows}), correct_queries=correct,
        covered_queries=covered, predicted_selection_accuracy=correct/n,
        conditional_selection_accuracy=correct/covered if covered else None,
        matched_candidate_coverage=covered/n,
        geometric_recall_at_0_5=sum(r['geometric_recalled'] for r in rows)/n,
        mean_target_mask_iou=sum(r['target_mask_iou'] for r in rows)/n,
        success_at_0_5=sum(r['target_mask_iou'] >= .5 for r in rows)/n,
        status_counts=dict(Counter(r['status'] for r in rows)))


def evaluate_saved_associations(queries, predictions, coverage, package, package_path, annotations):
    """Reuse the prior wkq association results; compute only new selection/IoU values."""
    qs, ps, cs = (unique_index(x, 'query_id') for x in (queries, predictions, coverage['rows']))
    require(qs and set(qs) == set(ps) == set(cs), 'Incomplete or mismatched denominator')
    require(len({p['run_id'] for p in predictions}) == 1 and
            {p['method_id'] for p in predictions} == {'B1'}, 'Mixed run/method')
    images = unique_index(package['images'], 'image_id')
    anns = unique_index(annotations, 'id')
    masks, targets, rows = {}, {}, []
    for qid, q in qs.items():
        p, old = ps[qid], cs[qid]
        require(q['split'] == 'dev', 'Dev only')
        require(all(p[k] == q[k] for k in ('image_id', 'query', 'language', 'split')), 'Query mismatch')
        require(old['image_id'] == q['image_id'] and old['target_instance_id'] == q['target_instance_id'],
                'Cached target mismatch')
        ann = anns[q['target_instance_id']]
        require(ann['image_id'] == q['image_id'] and not ann['iscrowd'], 'Invalid target')
        row = images[q['image_id']]
        candidates = unique_index(row['candidates'], 'candidate_id')
        matched = old['matched_candidate_id']
        require(old['covered'] == (matched is not None) and (matched is None or matched in candidates),
                'Invalid frozen association')
        cid = p['selected_candidate']['candidate_id'] if p['selected_candidate'] else None
        require((p['status'] == 'success') == (cid is not None), 'Invalid selected status')
        overlap = 0.
        if cid is not None:
            require(cid in candidates, 'Wrong selected image')
            c = candidates[cid]
            width, height = row['image_size_wh']
            if cid not in masks:
                path = safe_path(Path(package_path).parent, c['mask_reference'])
                require(sha256(path) == c['mask_sha256'], 'Selected mask changed')
                masks[cid] = decode_mask(read_json(path), height, width)
            if ann['id'] not in targets:
                targets[ann['id']] = decode_mask(ann['segmentation'], height, width)
            overlap = mask_iou(masks[cid], targets[ann['id']])
        rows.append(dict(query_id=qid, semantic_query_id=q['semantic_query_id'],
            image_id=q['image_id'], language=q['language'], category_id=ann['category_id'],
            status=p['status'], selected_candidate_id=cid, target_instance_id=ann['id'],
            correct=cid is not None and cid == matched, covered=old['covered'],
            geometric_recalled=old['geometric_recalled'], target_mask_iou=overlap))
    groups = {field: {str(value): summarize([r for r in rows if r[field] == value])
              for value in sorted({r[field] for r in rows})} for field in ('language', 'category_id')}
    result = summarize(rows)
    require(result['matched_candidate_coverage'] == coverage['summary']['matched_candidate_coverage'],
            'Coverage changed')
    return dict(evaluation_basis='user_authorized_development', summary=result, groups=groups, rows=rows,
        association_source='previous frozen coverage; not recomputed',
        note='lcy B1 integration measurement, not wyz model selection or final test')
