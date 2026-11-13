"""CPU-only paired diagnostics of immutable COCO predictions; never loads a model.

COCO matching is recomputed to expose per-instance/rank diagnostics absent from
the aggregate reports. Predictions and annotations are never modified. This is
not the language-candidate decoder or the query-weighted coverage evaluator.
"""
import argparse
from collections import Counter, defaultdict
import contextlib
import copy
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import time

import numpy as np
from pycocotools import mask as mask_api
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def valid_mean(values):
    values = np.asarray(values)
    values = values[values >= 0]
    return float(values.mean() * 100) if values.size else None


def quantiles(values):
    return [float(x) for x in np.quantile(values, [0, .25, .5, .75, 1])] if values else None


def analyze(coco, rows, report, high_score):
    # Stable flattening preserves the original order used for tied COCO scores.
    flat = [d for row in rows for d in row['detections']]
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = copy.deepcopy(coco)
        gt.dataset.setdefault('info', {})
        gt.createIndex()
        dt = gt.loadRes(copy.deepcopy(flat))
        ev = COCOeval(gt, dt, 'segm')
        ev.params.imgIds = sorted(gt.imgs)
        ev.params.catIds = list(range(1, 9))
        ev.params.maxDets = [1, 10, 100]
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    # A guard on the new diagnostic derivation, not a new model verification.
    for i, key in enumerate(('AP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large')):
        assert abs(float(ev.stats[i] * 100) - report['metrics'][key]) < 1e-8, key
    by_cat = {}
    matches = {}
    dt_match = {}
    dt_ignore = {}
    for e in ev.evalImgs:
        if e is None or e['aRng'] != ev.params.areaRng[0]:
            continue
        for j, gid in enumerate(e['gtIds']):
            if not e['gtIgnore'][j]:
                matches[gid] = {str(int(round(t * 100))): int(e['gtMatches'][i, j])
                                for i, t in enumerate(ev.params.iouThrs)}
        for j, did in enumerate(e['dtIds']):
            dt_match[did] = [int(e['dtMatches'][i, j]) for i in (0, 5)]
            dt_ignore[did] = [bool(e['dtIgnore'][i, j]) for i in (0, 5)]

    per_gt, per_detection = [], []
    for iid in sorted(gt.imgs):
        gs = [g for g in gt.imgToAnns[iid] if not g['iscrowd'] and not g.get('ignore', False)]
        ds = dt.imgToAnns[iid]
        grles = [gt.annToRLE(g) for g in gs]
        drles = [d['segmentation'] for d in ds]
        overlaps = np.asarray(mask_api.iou(drles, grles, [0] * len(gs))) if gs else np.zeros((len(ds), 0))
        dlabels = np.array([d['category_id'] for d in ds])
        for j, g in enumerate(gs):
            same = np.flatnonzero(dlabels == g['category_id'])
            other = np.flatnonzero(dlabels != g['category_id'])
            def best(indices):
                if not len(indices):
                    return {'iou': 0., 'detection_id': None, 'score': None, 'category_id': None}
                k = int(indices[np.argmax(overlaps[indices, j])])
                return dict(iou=float(overlaps[k, j]), detection_id=ds[k]['id'],
                            score=ds[k]['score'], category_id=ds[k]['category_id'])
            per_gt.append(dict(annotation_id=g['id'], image_id=iid, category_id=g['category_id'],
                               source_category_id=g.get('source_category_id'), area=g['area'],
                               small=g['area'] <= 32**2,
                               best_same=best(same), best_other=best(other),
                               matched_detection_ids=matches[g['id']]))
        glabels = np.array([g['category_id'] for g in gs])
        for k, d in enumerate(ds):
            same = overlaps[k, glabels == d['category_id']]
            other = overlaps[k, glabels != d['category_id']]
            same_max = float(same.max()) if same.size else 0.
            other_max = float(other.max()) if other.size else 0.
            kinds = {}
            for ti, threshold in enumerate((.5, .75)):
                if dt_ignore[d['id']][ti]:
                    kind = 'ignored'
                elif dt_match[d['id']][ti]:
                    kind = 'tp'
                elif same_max >= threshold:
                    kind = 'duplicate_same_class'
                elif other_max >= threshold:
                    kind = 'wrong_class_overlap'
                elif max(same_max, other_max) >= .1:
                    kind = 'localization_or_mixed_below_iou'
                else:
                    kind = 'background_or_unannotated'
                kinds[str(int(threshold * 100))] = kind
            per_detection.append(dict(detection_id=d['id'], image_id=iid, category_id=d['category_id'],
                                      score=d['score'], area=d['area'], best_same_iou=same_max,
                                      best_other_iou=other_max, error=kinds))

    precision, recall = ev.eval['precision'], ev.eval['recall']
    for k, cid in enumerate(ev.params.catIds):
        gs = [g for g in per_gt if g['category_id'] == cid]
        ds = sorted([d for d in per_detection if d['category_id'] == cid], key=lambda d: (-d['score'], d['detection_id']))
        by_cat[str(cid)] = {
            'gt_count': len(gs), 'positive_images': len({g['image_id'] for g in gs}),
            'AP': valid_mean(precision[:, :, k, 0, -1]),
            'AP50': valid_mean(precision[0, :, k, 0, -1]),
            'AP75': valid_mean(precision[5, :, k, 0, -1]),
            'AR': valid_mean(recall[:, k, 0, -1]),
            'recall50': valid_mean(recall[0, k, 0, -1]),
            'recall75': valid_mean(recall[5, k, 0, -1]),
            'sizes': {},
            'matched_gt50': sum(g['matched_detection_ids']['50'] > 0 for g in gs),
            'matched_gt75': sum(g['matched_detection_ids']['75'] > 0 for g in gs),
            'same_class_best_iou_mean': float(np.mean([g['best_same']['iou'] for g in gs])) if gs else None,
            'fp_at_score_ge_0_5': dict(Counter(d['error']['50'] for d in ds if d['score'] >= high_score)),
            'all_detection_outcomes50': dict(Counter(d['error']['50'] for d in ds)),
            'score_quantiles_tp50': quantiles([d['score'] for d in ds if d['error']['50'] == 'tp']),
            'score_quantiles_fp50': quantiles([d['score'] for d in ds if d['error']['50'] not in ('tp', 'ignored')]),
            'top20_outcomes50': dict(Counter(d['error']['50'] for d in ds[:20])),
            'highest_score_fp50': [d for d in ds if d['error']['50'] not in ('tp', 'ignored')][:8],
        }
        for a, label in enumerate(ev.params.areaRngLbl[1:], 1):
            low, high = ev.params.areaRng[a]
            subset = [g for g in gs if low <= g['area'] <= high]
            by_cat[str(cid)]['sizes'][label] = dict(gt_count=len(subset),
                AP=valid_mean(precision[:, :, k, a, -1]), AR=valid_mean(recall[:, k, a, -1]))
        assert abs(by_cat[str(cid)]['AP'] - report['metrics']['per_category_AP'][str(cid)]) < 1e-8
    return by_cat, per_gt, per_detection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--source-commit', required=True)
    args = parser.parse_args()
    start = time.monotonic()
    config = json.loads(Path(args.config).read_text())
    root = Path(args.root)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    write(output/'resolved_config.json', config)
    package = root/config['package']
    manifest_path, annotation_path = package/'image_manifest.json', package/'annotations_coco.json'
    assert digest(manifest_path) == config['image_manifest_sha256']
    assert digest(annotation_path) == config['annotation_sha256']
    manifest = json.loads(manifest_path.read_text())
    assert all(im['split'] == 'dev' for im in manifest['images'])
    coco = json.loads(annotation_path.read_text())
    records, details, detections, reports, inputs = {}, {}, {}, {}, {}
    for item in config['inputs']:
        u = str(item['update'])
        base = root/item['directory']
        report_path, prediction_path = base/'evaluation_report.json', base/'predictions.json'
        assert digest(report_path) == item['report_sha256']
        report = json.loads(report_path.read_text())
        assert digest(prediction_path) == report['predictions_sha256']
        assert report['evaluation_data'] == {k: config[k] for k in ('image_manifest_sha256', 'annotation_sha256')}
        assert report['completed_updates'] == item['update'] and report['complete'] and report['inference_errors'] == 0
        rows = json.loads(prediction_path.read_text())
        assert len(rows) == len(manifest['images']) and {r['image_id'] for r in rows} == {im['image_id'] for im in manifest['images']}
        assert all(r['status'] in ('ok', 'no_detections') for r in rows)
        records[u], details[u], detections[u] = analyze(coco, rows, report, config['diagnostic_high_score'])
        reports[u] = report
        inputs[u] = dict(predictions_path=str(prediction_path), predictions_sha256=digest(prediction_path),
                         report_path=str(report_path), report_sha256=digest(report_path),
                         checkpoint_path=report['checkpoint_path'], checkpoint_sha256=report['checkpoint_sha256'])
        for name, data in [('per_gt', details[u]), ('per_detection', detections[u])]:
            (output/f'{name}_{u}.jsonl').write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in data))
        print(json.dumps({'update': u, 'phase': 'offline_matching_complete', 'gt': len(details[u])}), flush=True)
    a, b = [str(x['update']) for x in config['inputs']]
    old, new = [{g['annotation_id']: g for g in details[u]} for u in (a, b)]
    assert old.keys() == new.keys()
    paired = []
    for gid, x in old.items():
        y = new[gid]
        paired.append(dict(annotation_id=gid, image_id=x['image_id'], category_id=x['category_id'],
            area=x['area'], small=x['small'], old=x, new=y,
            same_iou_delta=y['best_same']['iou']-x['best_same']['iou']))
    (output/'paired_gt.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in paired))
    groups = {}
    for name, subset in [('all', paired), ('skirt', [p for p in paired if p['category_id']==3]),
                         ('outerwear', [p for p in paired if p['category_id']==4]),
                         ('small', [p for p in paired if p['small']])]:
        g = dict(gt_count=len(subset), images=len({p['image_id'] for p in subset}),
                 category_counts=dict(Counter(str(p['category_id']) for p in subset)), transitions={})
        for threshold in ('50', '75'):
            trans = Counter(('retained' if x['old']['matched_detection_ids'][threshold] else 'gained')
                            if x['new']['matched_detection_ids'][threshold] else
                            ('lost' if x['old']['matched_detection_ids'][threshold] else 'miss_both') for x in subset)
            g['transitions'][threshold] = dict(trans)
        for u, key in ((a,'old'), (b,'new')):
            g[u] = dict(same_iou_mean=float(np.mean([p[key]['best_same']['iou'] for p in subset])) if subset else None,
                same_ge50=sum(p[key]['best_same']['iou']>=.5 for p in subset),
                any_ge50=sum(max(p[key]['best_same']['iou'],p[key]['best_other']['iou'])>=.5 for p in subset),
                wrong_class_only_ge50=sum(p[key]['best_same']['iou']<.5<=p[key]['best_other']['iou'] for p in subset))
        g['largest_same_iou_drops'] = sorted(subset, key=lambda p:(p['same_iou_delta'], p['annotation_id']))[:6]
        groups[name] = g
    summary = dict(schema_version=1, purpose='paired_saved_AP_prediction_diagnostics',
        parent_metrics={u:r['metrics'] for u,r in reports.items()}, per_category=records, groups=groups,
        methods=config['diagnostic_definitions'], inference_calls=0, optimizer_updates=0, gpu_used=False,
        official_report_agreement=True, production_candidate_package_created=False)
    write(output/'analysis.json', summary)
    assert 'torch' not in sys.modules and not any(k.startswith('detectron2') for k in sys.modules)
    meta = dict(record_schema_version=1, run_id=output.name, experiment_id='S0', owner='segmentation', executed_by='lcy',
        decision_executor='Codex', status='completed', git_commit=args.source_commit, split='dev', candidate_source='none',
        seed=42, source_script_sha256=digest(__file__), config_sha256=digest(args.config),
        parent_run_ids=['segmentation-resume5000-ap-20261113-r01'], inputs=inputs,
        command=[sys.executable, *sys.argv], elapsed_seconds=time.monotonic()-start,
        finished_at=datetime.now(timezone.utc).isoformat(), inference_calls=0, optimizer_updates=0,
        torch_imported=False, checkpoint_loaded=False, queries_read=False,
        artifacts={p.name:{'path':str(p),'sha256':digest(p)} for p in output.iterdir() if p.is_file()})
    write(output/'run_metadata.json', meta)
    print(json.dumps({'status':'completed','elapsed_seconds':meta['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
