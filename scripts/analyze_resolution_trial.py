"""New B800/B1024 diagnostics from saved masks only; reuse existing A ledgers."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from pycocotools import mask as mask_api
from analyze_saved_segmentation_predictions import analyze, digest, write
from analyze_small_segmentation_errors import geometry_bucket, extent, summarize


def inner_boundary(mask):
    """One original-image pixel inner band, 8-neighbour erosion, zero outside."""
    mask = np.asarray(mask, dtype=bool)
    padded = np.pad(mask, 1, constant_values=False)
    eroded = np.ones_like(mask)
    h, w = mask.shape
    for dy in range(3):
        for dx in range(3):
            eroded &= padded[dy:dy+h, dx:dx+w]
    return mask & ~eroded


def boundary_iou(gt, pred):
    a, b = inner_boundary(gt), inner_boundary(pred)
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum()/union) if union else 0.


def paired(old, new):
    a, b = ({r['annotation_id']: r for r in rows} for rows in (old, new))
    if a.keys() != b.keys() or len(a) != len(old) or len(b) != len(new):
        raise ValueError('Paired GT population differs or duplicates exist')
    for k in a:
        if any(a[k][f] != b[k][f] for f in ('image_id', 'category_id', 'area')):
            raise ValueError('Paired GT identity changed')
    out = {}
    for field in ('matched50', 'matched75'):
        gained = sorted(k for k in a if not a[k][field] and b[k][field])
        lost = sorted(k for k in a if a[k][field] and not b[k][field])
        out[field] = dict(before=sum(r[field] for r in old), after=sum(r[field] for r in new),
                          net=len(gained)-len(lost), gained=gained, lost=lost,
                          gained_categories=dict(Counter(str(b[k]['category_id']) for k in gained)),
                          lost_categories=dict(Counter(str(a[k]['category_id']) for k in lost)))
    common = sorted(k for k in a if a[k]['matched50'] and b[k]['matched50'])
    def means(ids, field):
        return dict(n=len(ids), before=float(np.mean([a[k][field] for k in ids])) if ids else None,
                    after=float(np.mean([b[k][field] for k in ids])) if ids else None)
    out['common_matched50_ids'] = common
    if all('inner_boundary_iou_1px' in r for r in old+new):
        out['boundary_all_small'] = means(sorted(a), 'inner_boundary_iou_1px')
        out['boundary_common_matched50'] = means(common, 'inner_boundary_iou_1px')
    for f in ('precision', 'recall', 'false_positive_pixels', 'false_negative_pixels'):
        out['common_matched50_'+f] = dict(n=len(common),
            before=float(np.mean([a[k]['extent'][f] for k in common])) if common else None,
            after=float(np.mean([b[k]['extent'][f] for k in common])) if common else None)
    return out


def diagnose(plan, base, output, label):
    started = time.monotonic()
    output.mkdir(exist_ok=False)
    report = json.loads((base/'evaluation_report.json').read_text())
    if not (report['complete'] and report['selection_eligible'] and report['inference_errors'] == 0
            and report['checkpoint_sha256'] == plan['checkpoint_sha256']
            and report['completed_updates'] == 20000 and report['optimizer_updates'] == 0
            and report['checkpoint_model_unchanged'] and report['queries_read'] is False
            and report['evaluation_data'] == plan['evaluation_data']):
        raise ValueError('Incomplete or wrong B evaluation')
    if digest(base/'predictions.json') != report['predictions_sha256']:
        raise ValueError('Prediction identity changed')
    rows = json.loads((base/'predictions.json').read_text())
    coco = json.loads((Path(plan['evaluation_package'])/'annotations_coco.json').read_text())
    if digest(Path(plan['evaluation_package'])/'annotations_coco.json') != plan['evaluation_data']['annotation_sha256']:
        raise ValueError('Annotation identity changed')
    if (len(rows) != 250 or len({r['image_id'] for r in rows}) != 250
            or {r['image_id'] for r in rows} != {im['id'] for im in coco['images']}
            or any(r['status'] not in ('ok', 'no_detections') for r in rows)):
        raise ValueError('Need exactly the complete dev250 prediction population')
    category, gt_records, _ = analyze(coco, rows, report, .5)
    anns = {g['id']: g for g in coco['annotations']}
    detections = {i+1: d for i, d in enumerate(d for r in rows for d in r['detections'])}
    by_image = {r['image_id']: r['detections'] for r in rows}
    small = []
    for g in gt_records:
        if not g['small']:
            continue
        ann = anns[g['annotation_id']]
        gm = mask_api.decode(ann['segmentation']).astype(bool)
        d = detections.get(g['best_same']['detection_id'])
        pm = mask_api.decode(d['segmentation']).astype(bool) if d else np.zeros_like(gm)
        sizes = extent(gm, pm)
        if abs(sizes['iou']-g['best_same']['iou']) > 1e-10:
            raise ValueError('Diagnostic mask association differs')
        ds = by_image[g['image_id']]
        ious = np.asarray(mask_api.iou([x['segmentation'] for x in ds], [ann['segmentation']], [0])).reshape(-1)
        ranked = sorted([(i, x) for i, x in enumerate(ds) if ious[i] >= .5], key=lambda p: (-p[1]['score'], p[0]))
        top = dict(category_id=ranked[0][1]['category_id'], score=ranked[0][1]['score'],
                   iou=float(ious[ranked[0][0]])) if ranked else None
        mid = g['matched_detection_ids']['50']
        small.append(dict(**g, source_category_name=ann['source_category_name'],
            geometry_bucket=geometry_bucket(g['best_same']['iou'], g['best_other']['iou']),
            matched50=bool(mid), matched75=bool(g['matched_detection_ids']['75']),
            matched50_score=detections[mid]['score'] if mid else None,
            any_ge50=bool(ranked), highest_score_overlap=top, extent=sizes,
            inner_boundary_iou_1px=boundary_iou(gm, pm)))
    write(output/'small_gt.json', small)
    (output/'per_gt.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in gt_records))
    summary = dict(label=label, all_small=summarize(small),
        by_category={str(k): summarize([r for r in small if r['category_id'] == k]) for k in sorted({r['category_id'] for r in small})},
        by_source_category={k: summarize([r for r in small if r['source_category_name'] == k]) for k in sorted({r['source_category_name'] for r in small})},
        by_area={name:summarize([r for r in small if lo < r['area'] <= hi]) for name,lo,hi in [('1_256',0,256),('257_512',256,512),('513_1024',512,1024)]},
        category_metrics=category, report_path=str(base/'evaluation_report.json'),
        report_sha256=digest(base/'evaluation_report.json'), predictions_sha256=report['predictions_sha256'],
        all_small_inner_boundary_iou_1px_mean=float(np.mean([r['inner_boundary_iou_1px'] for r in small])),
        boundary_definition='GT-wise independent best same-class mask; 1-original-pixel inner band, 8-neighbour erosion with zero exterior. Fixed thin-band descriptive IoU, NOT standard Boundary AP or a matched-only performance metric.',
        matched_definition='Existing COCO all-area greedy same-class score-ordered matches, then subset GT area<=1024; distinct from COCO area-specific AP-small.',
        per_gt_sha256=digest(output/'per_gt.jsonl'), small_gt_sha256=digest(output/'small_gt.json'),
        inference_calls=0, optimizer_updates=0, torch_imported='torch' in sys.modules,
        elapsed_seconds=time.monotonic()-started)
    write(output/'summary.json', summary)
    return report, small, summary


def compare(plan, output):
    base = Path(plan['baseline_directory'])
    a_report = json.loads(Path(plan['a_report']).read_text())
    old_report = json.loads(Path(plan['old_best_report']).read_text())
    b_report = json.loads((base/'evaluation_report.json').read_text())
    new_report, new_rows, new_summary = diagnose(plan, output/'dev1024/update_020000', output/'diagnostic1024', 'B20k_1024')
    new_config = json.loads((output/'dev1024/resolved_config.json').read_text())
    config_hash = hashlib.sha256(json.dumps(new_config, sort_keys=True, allow_nan=False).encode()).hexdigest()
    if new_report['evaluation_config_sha256'] != config_hash or (new_config['short_edge'],new_config['max_size']) != (1024,1706):
        raise ValueError('New report has the wrong resize configuration')
    for key in ('protocol','metric_units','training_config_sha256','training_git_commit','training_data'):
        if new_report[key] != b_report[key]:
            raise ValueError('Paired evaluation differs in '+key)
    b_rows = json.loads((output/'diagnostic800/small_gt.json').read_text())
    b_summary = json.loads((output/'diagnostic800/summary.json').read_text())
    a_rows = json.loads(Path(plan['a_small_gt']).read_text())
    transitions = paired(b_rows, new_rows)
    delta = {k:new_report['metrics'][k]-b_report['metrics'][k] for k in ('AP','AP50','AP75','AP_small','AP_medium','AP_large')}
    # Same B weights; only resolution is selected. Ties retain cheaper cached 800.
    chosen = 1024 if delta['AP'] > 0 else 800
    target_gain = max(transitions['matched50']['net'], transitions['matched75']['net']) >= 5
    weak_support = any(new_summary['by_category'][str(k)][f] > b_summary['by_category'][str(k)][f]
                       for k in (6,8) for f in ('matched50','matched75'))
    boundary = transitions['boundary_all_small']
    practical_gate = delta['AP'] >= 0 and target_gain and weak_support and boundary['after'] >= boundary['before']
    result = dict(status='completed', selected_resolution=chosen, checkpoint_sha256=plan['checkpoint_sha256'],
        selection_rule='Highest full dev250 mask AP, tie keeps cached 800; no weight changed',
        delta_1024_minus_800=delta,
        per_category_AP_delta={k:new_report['metrics']['per_category_AP'][k]-b_report['metrics']['per_category_AP'][k] for k in b_report['metrics']['per_category_AP']},
        metrics=dict(A800_reused=a_report['metrics'], old18k800_reused=old_report['metrics'], B800_reused=b_report['metrics'], B1024_new=new_report['metrics']),
        A_to_B800=paired(a_rows,b_rows), B800_to_B1024=transitions,
        practical_gate=dict(passed=practical_gate, overall_AP_nondecrease=delta['AP']>=0,
            small_net_gain_ge5=target_gain, shoes_or_accessory_match_support=weak_support,
            boundary_all_small_nondecrease=boundary['after']>=boundary['before'],
            meaning='Predeclared practical screen, not statistical significance or an automatic training authorization'),
        eval_elapsed_seconds=dict(B800_reused=b_report['elapsed_seconds'],B1024=new_report['elapsed_seconds']),
        eval_peak_allocated_bytes=dict(B800_reused=b_report['peak_cuda_allocated_bytes'],B1024=new_report['peak_cuda_allocated_bytes']),
        new_inference_passes=1, training_updates=0, smoke_runs=0, candidates_exported=False,
        warnings=['Single fixed dev set, no new seed; no claim of statistical significance',
                  'Existing u18000 language candidates/caches remain unchanged',
                  'Historical A per-GT masks reused; no A inference/matching repeated',
                  'Boundary association uses best same-class IoU for diagnosis, not GT-assisted inference'])
    write(output/'comparison.json', result)
    print(json.dumps(dict(selected_resolution=chosen, delta=delta, practical_gate=result['practical_gate'])), flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',required=True); p.add_argument('--output',required=True)
    p.add_argument('--phase',choices=['baseline','compare'],required=True)
    args=p.parse_args(); plan=json.loads(Path(args.plan).read_text()); out=Path(args.output)
    if args.phase=='baseline':
        _,_,s=diagnose(plan,Path(plan['baseline_directory']),out/'diagnostic800','B20k_800')
        print(json.dumps(s['all_small']),flush=True)
    else:
        compare(plan,out)

if __name__=='__main__':main()
