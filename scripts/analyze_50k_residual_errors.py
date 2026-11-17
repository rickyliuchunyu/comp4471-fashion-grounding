"""CPU diagnostics from frozen 50k AP predictions and candidate cache only.

Reuse existing geometry/extent/boundary definitions; do not recompute AP or
modify production matching, candidates, thresholds, labels or checkpoints.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from pycocotools import mask as mask_api
from analyze_saved_segmentation_predictions import digest, quantiles, write
from analyze_small_segmentation_errors import geometry_bucket, extent
from analyze_resolution_trial import boundary_iou


def matrix_associate(candidates, annotations, overlaps, threshold=.5):
    """Same score/ID and IoU/GT-ID tie rules as wkq associate; RLE IoU input."""
    available = set(range(len(annotations)))
    result = {}
    for k in sorted(range(len(candidates)), key=lambda k: (-candidates[k]['score'], candidates[k]['candidate_id'])):
        options = [j for j in available if not annotations[j]['iscrowd'] and
                   annotations[j]['category_id'] == candidates[k]['category_id']]
        if not options:
            continue
        j = min(options, key=lambda j: (-overlaps[k, j], annotations[j]['id']))
        if overlaps[k, j] >= threshold:
            result[candidates[k]['candidate_id']] = annotations[j]['id']
            available.remove(j)
    return result


def residual_cause(selected, raw, matched):
    if matched:
        return 'matched50'
    if selected['best_same']['iou'] >= .5:
        return 'one_to_one_contention'
    if raw['best_same']['iou'] >= .5:
        return 'score_filtered'
    if raw['best_other']['iou'] >= .5:
        return 'wrong_class_only'
    if max(raw['best_same']['iou'], raw['best_other']['iou']) >= .1:
        return 'partial_or_mixed_geometry'
    return 'no_saved_mask_iou_ge_010'


def normalized_rle(rle):
    if isinstance(rle['counts'], list):
        return mask_api.frPyObjects(rle, *rle['size'])
    return rle


def view(candidates, overlaps, j, category):
    def best(same):
        ids = [k for k,c in enumerate(candidates) if (c['category_id'] == category) == same]
        if not ids:
            return dict(iou=0., candidate_id=None, category_id=None, score=None)
        k = max(ids, key=lambda k: overlaps[k,j])
        return dict(iou=float(overlaps[k,j]), candidate_id=candidates[k]['candidate_id'],
                    category_id=candidates[k]['category_id'], score=candidates[k]['score'])
    a,b = best(True),best(False)
    return dict(best_same=a,best_other=b,geometry_bucket=geometry_bucket(a['iou'],b['iou']))


def summarize(rows):
    def mean(values):
        return float(np.mean(values)) if values else None
    out = dict(gt_count=len(rows),images=len({r['image_id'] for r in rows}),
        residual_causes=dict(Counter(r['residual_cause'] for r in rows)),
        matched50=sum(r['matched50'] for r in rows),matched75=sum(r['matched75'] for r in rows),
        matched50_iou_below75=sum(r['matched50'] and r['matched_extent']['iou'] < .75 for r in rows),
        ap_only_same_class_geometry50=sum(r['ap']['best_same']['iou'] >= .5 and r['raw']['best_same']['iou'] < .5 for r in rows),
        views={name:dict(geometry_partition=dict(Counter(r[name]['geometry_bucket'] for r in rows)),
            same_class_best50=sum(r[name]['best_same']['iou']>=.5 for r in rows),
            same_class_best75=sum(r[name]['best_same']['iou']>=.75 for r in rows),
            any_class_best50=sum(max(r[name]['best_same']['iou'],r[name]['best_other']['iou'])>=.5 for r in rows))
            for name in ('ap','raw','selected')})
    matched=[r for r in rows if r['matched50']]
    out['matched50_shape']=dict(n=len(matched),
        mean_precision=mean([r['matched_extent']['precision'] for r in matched]),
        mean_recall=mean([r['matched_extent']['recall'] for r in matched]),
        mean_inner_boundary_iou_1px=mean([r['matched_boundary_iou_1px'] for r in matched]),
        mean_iou=mean([r['matched_extent']['iou'] for r in matched]),
        recall_below08=sum(r['matched_extent']['recall']<.8 for r in matched),
        precision_below08=sum(r['matched_extent']['precision']<.8 for r in matched))
    out['score_filtered_best_same_scores']=quantiles([r['raw']['best_same']['score'] for r in rows if r['residual_cause']=='score_filtered'])
    out['wrong_class_destinations']=dict(Counter(str(r['category_id'])+'->'+str(r['raw']['best_other']['category_id']) for r in rows if r['residual_cause']=='wrong_class_only'))
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True);parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();started=time.monotonic();root=Path(args.root);output=Path(args.output)
    config=json.loads(Path(args.config).read_text());assert not output.exists()
    for relative,expected in config['input_sha256'].items():
        assert digest(root/relative)==expected, relative
    coco=json.loads((root/config['annotations']).read_text());images={im['id']:im for im in coco['images']}
    gt_by_image=defaultdict(list)
    for g in coco['annotations']:
        if not g['iscrowd'] and not g.get('ignore'):
            gt_by_image[g['image_id']].append(g)
    predictions={r['image_id']:r for r in json.loads((root/config['ap_predictions']).read_text())}
    cache_root=root/config['cache'];cache=json.loads((cache_root/'cache.json').read_text())
    cache_refs={r['image_id']:r for r in cache['records']}
    selected_package=json.loads((root/config['selected_package']).read_text())
    selected_rows={r['image_id']:r for r in selected_package['images']}
    coverage=json.loads((root/config['prior_coverage']).read_text())
    assert set(predictions)==set(images)==set(cache_refs)==set(selected_rows) and len(images)==250
    assert sum(map(len,gt_by_image.values()))==1008
    output.mkdir();write(output/'resolved_config.json',config)
    rows=[];detections=[];cached_associations_reused=0
    for number,iid in enumerate(sorted(images),1):
        assert time.monotonic()-started<config['budget_seconds']
        gs=gt_by_image[iid];ref=cache_refs[iid];path=cache_root/ref['path']
        assert digest(path)==ref['sha256']
        raw=json.loads(path.read_text())['candidates']
        raw_rles=[normalized_rle(json.loads((cache_root/c['mask_reference']).read_text())) for c in raw]
        gt_rles=[normalized_rle(g['segmentation']) for g in gs]
        raw_iou=np.asarray(mask_api.iou(raw_rles,gt_rles,[0]*len(gs))) if raw and gs else np.zeros((len(raw),len(gs)))
        selected=selected_rows[iid]['candidates'];raw_ids={c['candidate_id']:k for k,c in enumerate(raw)}
        selected_indexes=[raw_ids[c['candidate_id']] for c in selected]
        selected_iou=raw_iou[selected_indexes,:]
        match50=matrix_associate(selected,gs,selected_iou,.5)
        match75=matrix_associate(selected,gs,selected_iou,.75)
        if str(iid) in coverage['association']:
            assert match50==coverage['association'][str(iid)], 'New diagnostic association differs from existing result'
            cached_associations_reused+=1
        inverse50={aid:cid for cid,aid in match50.items()};inverse75={aid:cid for cid,aid in match75.items()}
        aps=[dict(d,candidate_id=f'ap-row-{k:03d}') for k,d in enumerate(predictions[iid]['detections'])]
        ap_iou=np.asarray(mask_api.iou([normalized_rle(d['segmentation']) for d in aps],gt_rles,[0]*len(gs))) if aps and gs else np.zeros((len(aps),len(gs)))
        for j,g in enumerate(gs):
            aa,rr,ss=(view(cs,ov,j,g['category_id']) for cs,ov in ((aps,ap_iou),(raw,raw_iou),(selected,selected_iou)))
            matched=g['id'] in inverse50
            metrics=None;boundary=None
            if matched:
                k=raw_ids[inverse50[g['id']]]
                gm=mask_api.decode(gt_rles[j]).astype(bool);pm=mask_api.decode(raw_rles[k]).astype(bool)
                metrics=extent(gm,pm);boundary=boundary_iou(gm,pm)
                assert abs(metrics['iou']-raw_iou[k,j])<1e-9
            scale=min(800/min(images[iid]['height'],images[iid]['width']),1333/max(images[iid]['height'],images[iid]['width']))
            rows.append(dict(annotation_id=g['id'],image_id=iid,category_id=g['category_id'],source_category=g['source_category_name'],
                area=g['area'],size='small' if g['area']<=1024 else 'medium' if g['area']<=9216 else 'large',
                bbox_xywh=g['bbox'],eval_scaled_bbox_wh=[g['bbox'][2]*scale,g['bbox'][3]*scale],
                ap=aa,raw=rr,selected=ss,matched50=matched,matched75=g['id'] in inverse75,
                matched50_candidate_id=inverse50.get(g['id']),matched_extent=metrics,matched_boundary_iou_1px=boundary,
                residual_cause=residual_cause(ss,rr,matched)))
        for k,c in enumerate(selected):
            same=[j for j,g in enumerate(gs) if g['category_id']==c['category_id']]
            other=[j for j,g in enumerate(gs) if g['category_id']!=c['category_id']]
            a=max((float(selected_iou[k,j]) for j in same),default=0.);b=max((float(selected_iou[k,j]) for j in other),default=0.)
            kind=('tp' if c['candidate_id'] in match50 else 'duplicate_same_class' if a>=.5 else
                  'wrong_class_overlap' if b>=.5 else 'partial_or_mixed_geometry' if max(a,b)>=.1 else 'background_or_unannotated')
            detections.append(dict(image_id=iid,candidate_id=c['candidate_id'],category_id=c['category_id'],score=c['score'],error=kind,best_same_iou=a,best_other_iou=b))
        if number%50==0:print(json.dumps(dict(images=number,gt=len(rows))),flush=True)
    groups={'all':rows,'small':[r for r in rows if r['size']=='small']}
    groups.update({f'category_{cid}':[r for r in rows if r['category_id']==cid] for cid in range(1,9)})
    groups.update({f'small_category_{cid}':[r for r in rows if r['size']=='small' and r['category_id']==cid] for cid in (6,7,8)})
    groups.update({f'small_area_{lo}_{hi}':[r for r in rows if lo<r['area']<=hi] for lo,hi in ((0,256),(256,512),(512,1024))})
    summaries={name:summarize(rs) for name,rs in groups.items()}
    fine={name:summarize([r for r in rows if r['source_category']==name]) for name in sorted({r['source_category'] for r in rows if r['category_id'] in (6,7,8)})}
    by_id={r['annotation_id']:r for r in rows}
    uncovered=[r for r in coverage['rows'] if not r['covered']]
    query_causes=dict(Counter(by_id[r['target_instance_id']]['residual_cause'] for r in uncovered))
    cases=[];used=set()
    # Deliberately chosen failure illustrations; numerical results use all GT.
    for cid in (6,7,8):
        for kind in ('missing','boundary'):
            possible=[r for r in rows if r['category_id']==cid and r['annotation_id'] not in used and
                ((not r['matched50']) if kind=='missing' else r['matched50'] and r['matched_extent']['iou']<.75)]
            possible.sort(key=lambda r:(r['size']!='small',r['raw']['best_same']['iou'] if kind=='missing' else r['matched_extent']['iou'],r['annotation_id']))
            if possible:
                chosen=possible[0];used.add(chosen['annotation_id']);cases.append(dict(kind=kind,**chosen))
    for cause in ('score_filtered','wrong_class_only'):
        possible=[r for r in rows if r['residual_cause']==cause and r['annotation_id'] not in used]
        possible.sort(key=lambda r:(r['size']!='small',-r['raw']['best_other' if cause=='wrong_class_only' else 'best_same']['iou'],r['annotation_id']))
        if possible:cases.append(dict(kind=cause,**possible[0]));used.add(possible[0]['annotation_id'])
    write(output/'per_gt.json',rows);write(output/'per_detection.json',detections);write(output/'selected_cases.json',cases)
    summary=dict(status='completed',groups=summaries,weak_source_categories=fine,
        selected_detection_errors=dict(Counter(d['error'] for d in detections)),
        selected_detection_errors_by_category={str(cid):dict(Counter(d['error'] for d in detections if d['category_id']==cid)) for cid in range(1,9)},
        existing_uncovered_queries=dict(query_records=len(uncovered),unique_gt=len({r['target_instance_id'] for r in uncovered}),causes=query_causes),
        reused_ap_metrics=json.loads((root/config['ap_report']).read_text())['metrics'],
        existing_selected_association_images_verified=cached_associations_reused,
        definitions=dict(population='All 1008 noncrowd/nonignored dev250 GT; small original mask area <=1024; medium 1024<area<=9216.',
            geometry='Independent best same/other-class mask IoU; not one-to-one. AP top100 and one-class-per-query cache are separate views. AP row IDs are local ordinals, never decoder query IDs.',
            residual_cause='Exclusive hierarchy for selected .3: matched50; same-class selected geometry>=.5 but consumed by another GT; correct raw geometry>=.5 filtered by score; only wrong-class raw geometry>=.5; any raw geometry>=.1; no raw geometry>=.1.',
            matching='Selected .3 score/ID ordered same-class greedy one-to-one; IoU tie smallest GT ID, identical to wkq association at .5. .75 uses the same diagnostic rule, not COCO AP/AR.',
            shape='Pixel precision/recall and 1-original-pixel inner-boundary IoU on actual .5 matched pairs. Neither standard Boundary AP nor causal AP-loss decomposition; matched subset only.',
            interpretation='Score is class probability times mask quality. Cache lacks original logits, so score-filtered loss is not proof of classification alone. No saved mask is an output-level observation, not proof of backbone cause.',
            cases='Deterministic failure illustrations selected by category/size/low IoU; not a random quality sample.'),
        inference_calls=0,checkpoint_loads=0,optimizer_updates=0,ap_recomputations=0,smoke_reruns=0,
        torch_imported='torch' in sys.modules,elapsed_seconds=time.monotonic()-started,
        finished_at=datetime.now(timezone.utc).isoformat(),source_script_sha256=digest(__file__),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[1],text=True).strip())
    assert not summary['torch_imported'] and cached_associations_reused==246
    assert sum(summary['selected_detection_errors'].values())==1279
    write(output/'summary.json',summary)
    print(json.dumps(dict(status='completed',elapsed_seconds=summary['elapsed_seconds'],small=summaries['small'],query_causes=query_causes)))


if __name__=='__main__':main()
