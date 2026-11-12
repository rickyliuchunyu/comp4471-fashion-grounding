"""CPU-only diagnosis of saved outputs and logged augmentation shapes; no model."""
import collections
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from detectron2.data import transforms as T
from pycocotools import mask as M

ROOT = Path('/root/autodl-tmp/fashion-grounding')
OUT = ROOT / 'runs/segmentation/low-ap-diagnosis-20261112-r01'
OUT.mkdir(exist_ok=False)
started = time.monotonic()
train_run = ROOT / 'runs/segmentation/segmentation-real-dev-train-20261112-r01'
eval_run = ROOT / 'runs/segmentation/segmentation-real-dev-ap-20261112-r01'
train = json.loads((ROOT/'shared/processed/fp_train5000_v1/annotations_coco.json').read_text())
dev = json.loads((ROOT/'shared/processed/fp_dev250_v1/annotations_coco.json').read_text())
logs = [json.loads(s) for s in (train_run/'metrics.jsonl').read_text().splitlines()]
cfg = json.loads((train_run/'resolved_config.json').read_text())
images = {x['id']: x for x in train['images']}
anns = collections.defaultdict(list)
for a in train['annotations']:
    if not a.get('iscrowd', 0): anns[a['image_id']].append(a)
np.random.seed(cfg['seed'])
replay = []
sample_categories = collections.Counter()
for log in logs:
    for iid, logged_count in zip(log['image_ids'], log['instance_counts']):
        im = images[iid]
        # Augmentation randomness depends on shape, not RGB pixel content.
        blank = np.zeros((im['height'], im['width'], 3), dtype=np.uint8)
        aug = [T.RandomFlip(horizontal=True, vertical=False),
               T.ResizeScale(cfg['min_scale'], cfg['max_scale'], cfg['image_size'], cfg['image_size']),
               T.FixedSizeCrop(crop_size=(cfg['image_size'], cfg['image_size']))]
        _, tr = T.apply_transform_gens(aug, blank)
        pad = tr.apply_segmentation(np.zeros(blank.shape[:2], dtype=np.uint8)) != 0
        old_count = corrected_count = ghosts = old_fg = added_fg = 0
        for a in sorted(anns[iid], key=lambda a: a['id']):
            sample_categories[a['category_id']] += 1
            bad = tr.apply_segmentation(M.decode(a['segmentation'])).astype(bool)
            good = bad & ~pad
            old_count += int(bad.any()); corrected_count += int(good.any())
            ghosts += int(bad.any() and not good.any())
            old_fg += int(bad.sum()); added_fg += int((bad & pad).sum())
        replay.append(dict(update=log['update'], image_id=iid, padding_pixels=int(pad.sum()),
                           padding_fraction=float(pad.mean()), logged_instances=logged_count,
                           replay_old_instances=old_count, corrected_instances=corrected_count,
                           ghost_instances=ghosts, wrong_foreground_pixels=added_fg,
                           old_foreground_pixels=old_fg))
(OUT/'augmentation_replay.json').write_text(json.dumps(replay,indent=2)+'\n')
replay_summary = dict(images=len(replay), images_with_padding=sum(x['padding_pixels']>0 for x in replay),
    old_instance_counts_match_all_logs=all(x['logged_instances']==x['replay_old_instances'] for x in replay),
    old_instances=sum(x['replay_old_instances'] for x in replay),
    corrected_instances=sum(x['corrected_instances'] for x in replay),
    ghost_instances=sum(x['ghost_instances'] for x in replay),
    median_padding_fraction=float(np.median([x['padding_fraction'] for x in replay])),
    wrong_foreground_fraction=sum(x['wrong_foreground_pixels'] for x in replay)/sum(x['old_foreground_pixels'] for x in replay),
    source_instances_by_category=dict(sample_categories),
    replay_basis='seed42 NumPy sequence, logged image order and dimensions; shape-only replay, no RGB decoding/model')
gt = collections.defaultdict(list)
for a in dev['annotations']:
    if not a.get('iscrowd',0): gt[a['image_id']].append(a)
def quantiles(values):
    return dict(zip(('min','p10','median','p90','max'),map(float,np.quantile(values,[0,.1,.5,.9,1]))))
pred_summary = {}
for update in (50,100):
    path = eval_run/f'update_{update:06d}/predictions.json'
    rows = json.loads(path.read_text())
    any_best, same_best, all_scores, unique_masks, pred_areas = [], [], [], [], []
    class_counts=collections.Counter(); confusion=np.zeros((8,8),dtype=int)
    total_masks = empty_masks = 0
    by_category=collections.defaultdict(lambda: {'gt':0,'any_iou50':0,'same_iou50':0})
    for row in rows:
        ds = row['detections']; aa=gt[row['image_id']]
        rles=[d['segmentation'] for d in ds]; areas=M.area(rles)
        total_masks += len(ds); empty_masks += int((areas==0).sum())
        all_scores.extend(d['score'] for d in ds);class_counts.update(d['category_id'] for d in ds)
        unique_masks.append(len({(tuple(d['segmentation']['size']),d['segmentation']['counts']) for d in ds}))
        pred_areas.extend(float(area)/(r['size'][0]*r['size'][1]) for area,r in zip(areas,rles))
        overlaps=M.iou(rles,[a['segmentation'] for a in aa],[0]*len(aa))
        labels=np.array([d['category_id'] for d in ds])
        for j,a in enumerate(aa):
            best=float(overlaps[:,j].max())
            same=overlaps[labels==a['category_id'],j]
            correct=float(same.max()) if same.size else 0.
            any_best.append(best);same_best.append(correct)
            c=by_category[a['category_id']];c['gt']+=1;c['any_iou50']+=int(best>=.5);c['same_iou50']+=int(correct>=.5)
            if best>=.5:
                # Among masks at maximal IoU, use highest score to break identical-mask class duplicates.
                candidates=np.flatnonzero(overlaps[:,j]>=best-1e-10)
                k=max(candidates,key=lambda k: ds[k]['score'])
                confusion[a['category_id']-1,ds[k]['category_id']-1]+=1
    pred_summary[str(update)] = dict(predictions_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        gt_instances=len(any_best), total_detections=total_masks, empty_masks=empty_masks,
        category_agnostic_gt_coverage_at_50=sum(x>=.5 for x in any_best)/len(any_best),
        same_category_gt_coverage_at_50=sum(x>=.5 for x in same_best)/len(same_best),
        category_agnostic_gt_coverage_at_75=sum(x>=.75 for x in any_best)/len(any_best),
        same_category_gt_coverage_at_75=sum(x>=.75 for x in same_best)/len(same_best),
        scores=quantiles(all_scores), prediction_area_fraction=quantiles(pred_areas),
        unique_masks_per_image=quantiles(unique_masks), predictions_by_category=dict(class_counts),
        best_geometry_predicted_class_confusion=confusion.tolist(),by_category=dict(by_category),
        scope='Existing top100 only. Coverage is per-GT any-match, not one-to-one recall/AP. No relabeling or threshold selection.')
report=dict(scope='CPU diagnosis only; no checkpoint loading, inference, COCOeval rerun or training',
    source_commit='b02f6e9d05b6a59297b64a4ad78494db45c75dd4',augmentation=replay_summary,predictions=pred_summary,
    elapsed_seconds=time.monotonic()-started)
(OUT/'diagnosis.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
