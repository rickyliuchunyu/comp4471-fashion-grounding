"""Explain saved small-object masks: no model, new inference, or metric editing.

Consumes the existing paired-analysis ledgers. A GT-level descriptive partition
is not a TIDE counterfactual AP contribution or language-query coverage.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from pycocotools import mask as mask_api

from analyze_saved_segmentation_predictions import digest, quantiles, write


def geometry_bucket(same, other):
    if same >= .75:
        return 'same_iou_ge_075'
    if same >= .5:
        return 'same_iou_050_075'
    if other >= .5:
        return 'wrong_class_only_iou_ge_050'
    if max(same, other) >= .1:
        return 'partial_or_mixed_iou_010_050'
    return 'no_saved_mask_iou_ge_010'


def extent(gt, pred):
    """Pixel coverage for a selected mask, not a contour distance metric."""
    assert gt.shape == pred.shape and gt.any()
    inter = int(np.logical_and(gt, pred).sum())
    pred_area, gt_area = int(pred.sum()), int(gt.sum())
    union = pred_area + gt_area - inter
    return dict(iou=inter/union, precision=inter/pred_area if pred_area else None,
                recall=inter/gt_area, false_negative_pixels=gt_area-inter,
                false_positive_pixels=pred_area-inter, predicted_area=pred_area,
                gt_area=gt_area)


def summarize(rows):
    if not rows:
        return dict(gt_count=0, images=0)
    return dict(gt_count=len(rows), images=len({r['image_id'] for r in rows}),
        geometry_partition=dict(Counter(r['geometry_bucket'] for r in rows)),
        matched50=sum(r['matched50'] for r in rows), matched75=sum(r['matched75'] for r in rows),
        matched50_score_ge_05=sum(r['matched50'] and r['matched50_score'] >= .5 for r in rows),
        matched50_score_quantiles=quantiles([r['matched50_score'] for r in rows if r['matched50']]),
        any_geometry_ge50=sum(r['any_ge50'] for r in rows),
        top_scored_ge50_wrong_class=sum(r['highest_score_overlap'] is not None and
            r['highest_score_overlap']['category_id'] != r['category_id'] for r in rows),
        same_best_iou_quantiles=quantiles([r['best_same']['iou'] for r in rows]),
        same_ge50_precision_quantiles=quantiles([r['extent']['precision'] for r in rows if r['best_same']['iou']>=.5]),
        same_ge50_recall_quantiles=quantiles([r['extent']['recall'] for r in rows if r['best_same']['iou']>=.5]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis',required=True)
    parser.add_argument('--root',required=True)
    args=parser.parse_args(); start=time.monotonic()
    analysis,root=Path(args.analysis),Path(args.root)
    config=json.loads((analysis/'resolved_config.json').read_text())
    meta=json.loads((analysis/'run_metadata.json').read_text())
    coco=json.loads((root/config['package']/'annotations_coco.json').read_text())
    assert digest(root/config['package']/'annotations_coco.json')==config['annotation_sha256']
    anns={g['id']:g for g in coco['annotations'] if not g.get('iscrowd') and not g.get('ignore')}
    ims={i['id']:i for i in coco['images']}; outputs={}; summaries={}
    assert not (analysis/'small_errors.json').exists(), 'Never overwrite prior diagnostic output'
    for item in config['inputs']:
        u=str(item['update']); predpath=root/item['directory']/'predictions.json'
        assert digest(predpath)==meta['inputs'][u]['predictions_sha256']
        prediction_rows=json.loads(predpath.read_text())
        flat=[d for r in prediction_rows for d in r['detections']]
        detections={i+1:d for i,d in enumerate(flat)}
        by_image={r['image_id']:r['detections'] for r in prediction_rows}
        gt_records=[json.loads(s) for s in (analysis/f'per_gt_{u}.jsonl').read_text().splitlines()]
        result=[]
        for g in gt_records:
            if not g['small']:continue
            ann=anns[g['annotation_id']];im=ims[g['image_id']]
            gm=mask_api.decode(ann['segmentation']).astype(bool)
            d=detections.get(g['best_same']['detection_id'])
            pm=mask_api.decode(d['segmentation']).astype(bool) if d else np.zeros_like(gm)
            sizes=extent(gm,pm)
            assert abs(sizes['iou']-g['best_same']['iou'])<1e-10
            ds=by_image[g['image_id']]
            ious=np.asarray(mask_api.iou([x['segmentation'] for x in ds],[ann['segmentation']],[0])).reshape(-1)
            # Highest score among masks actually overlapping this GT at >=.5.
            candidates=[(i,x) for i,x in enumerate(ds) if ious[i]>=.5]
            ranked=sorted(candidates,key=lambda pair:(-pair[1]['score'],pair[0]))
            top=(dict(category_id=ranked[0][1]['category_id'],score=ranked[0][1]['score'],
                      iou=float(ious[ranked[0][0]])) if ranked else None)
            matched_id=g['matched_detection_ids']['50']
            scale=min(800/min(im['height'],im['width']),1333/max(im['height'],im['width']))
            result.append(dict(**g, source_category_name=ann['source_category_name'],
                geometry_bucket=geometry_bucket(g['best_same']['iou'],g['best_other']['iou']),
                matched50=bool(matched_id), matched75=bool(g['matched_detection_ids']['75']),
                matched50_score=detections[matched_id]['score'] if matched_id else None,
                any_ge50=bool(ranked), highest_score_overlap=top, extent=sizes,
                gt_bbox_xywh=ann['bbox'],gt_bbox_fill=ann['area']/(ann['bbox'][2]*ann['bbox'][3]),
                approximate_eval_scaled_bbox_wh=[ann['bbox'][2]*scale,ann['bbox'][3]*scale]))
        assert len(result)==len([g for g in anns.values() if g['area']<=1024])
        outputs[u]=result
        summaries[u]=dict(all_small=summarize(result),
            by_category={str(k):summarize([r for r in result if r['category_id']==k]) for k in sorted({r['category_id'] for r in result})},
            by_source_category={k:summarize([r for r in result if r['source_category_name']==k]) for k in sorted({r['source_category_name'] for r in result})},
            by_area={k:summarize([r for r in result if low<r['area']<=high]) for k,low,high in [('1_256',0,256),('257_512',256,512),('513_1024',512,1024)]},
            wrong_class_only_destinations=dict(Counter(str(r['category_id'])+'->'+str(r['best_other']['category_id']) for r in result if r['geometry_bucket']=='wrong_class_only_iou_ge_050')))
        write(analysis/f'small_gt_{u}.json',result)
    updates=[str(i['update']) for i in config['inputs']]
    old,new=[{r['annotation_id']:r for r in outputs[u]} for u in updates]
    assert old.keys()==new.keys()
    transitions=Counter(old[k]['geometry_bucket']+' -> '+new[k]['geometry_bucket'] for k in old)
    # Deterministic, deliberately selected examples. Not a random quality sample.
    selected=[]; used=set()
    choices=[('no_overlap',lambda k:old[k]['geometry_bucket']==new[k]['geometry_bucket']=='no_saved_mask_iou_ge_010',lambda k:(-old[k]['area'],k)),
             ('wrong_class',lambda k:old[k]['geometry_bucket']=='wrong_class_only_iou_ge_050',lambda k:(-old[k]['best_other']['iou'],k)),
             ('partial_mask',lambda k:old[k]['geometry_bucket']==new[k]['geometry_bucket']=='partial_or_mixed_iou_010_050',lambda k:(-old[k]['best_same']['iou'],k)),
             ('boundary_limit',lambda k:old[k]['geometry_bucket']==new[k]['geometry_bucket']=='same_iou_050_075',lambda k:(-old[k]['best_same']['iou'],k)),
             ('largest_drop',lambda k:True,lambda k:(new[k]['best_same']['iou']-old[k]['best_same']['iou'],k)),
             ('largest_gain',lambda k:True,lambda k:(old[k]['best_same']['iou']-new[k]['best_same']['iou'],k))]
    for name,predicate,order in choices:
        candidates=sorted([k for k in old if k not in used and predicate(k)],key=order)
        if candidates:
            k=candidates[0];used.add(k);selected.append(dict(kind=name,annotation_id=k,image_id=old[k]['image_id'],old=old[k],new=new[k]))
    result=dict(schema_version=1,updates=updates,metrics_are='descriptive GT-level saved-mask diagnostics, not causal AP gains',
        methods=dict(small='noncrowd/nonignored original GT mask area <=1024 pixels; keep all small GT',
            partition='Independent best same/other-class IoU among saved official top100. First same>=.75, then same>=.5, then other>=.5, then any>=.1, otherwise no overlap. Exclusive and exhaustive, not one-to-one.',
            matched='Existing COCO all-area per-class greedy score-ordered matches, filtered to small GT afterwards; distinct from area-specific COCO AP-small.',
            extent='For independent max-IoU SAME-class saved mask: intersection/pred-area and intersection/GT-area. Shape/extent errors, not contour metric or proven feature-resolution cause.',
            scores='Fixed score .5 diagnostic slice only; not a selected production threshold. Score combines class and mask quality; no separate logits or stable query IDs exist here.',
            scale='Approximate bbox scaling from frozen eval short_edge800/max_size1333; no inference or train augmentation simulation.',
            cases='One deterministic deliberately chosen example per failure/change mode; not representative performance sampling.'),
        summaries=summaries,geometry_transitions=dict(transitions),selected_cases=selected,
        inference_calls=0,optimizer_updates=0,checkpoint_loaded=False,torch_imported='torch' in sys.modules,
        elapsed_seconds=time.monotonic()-start,finished_at=datetime.now(timezone.utc).isoformat(),
        source_script_sha256=digest(__file__),parent_analysis_sha256=digest(analysis/'analysis.json'),
        output_ledgers={u:digest(analysis/f'small_gt_{u}.json') for u in updates})
    assert not result['torch_imported']
    write(analysis/'small_errors.json',result)
    print(json.dumps(dict(summary=summaries,geometry_transitions=dict(transitions),cases=[{'kind':c['kind'],'annotation_id':c['annotation_id'],'image_id':c['image_id']} for c in selected]),indent=2))


if __name__=='__main__':main()
