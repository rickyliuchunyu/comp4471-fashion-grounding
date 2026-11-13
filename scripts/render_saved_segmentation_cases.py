"""Render selected paired error cases and explain FP overlaps without inference.

Reads the already-computed diagnostic ledger. GT-based crops/overlays are for
inspection only and are never exported as language candidates.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pycocotools import mask as mask_api


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--analysis', required=True)
    args = p.parse_args()
    root, analysis = Path(args.root), Path(args.analysis)
    output = analysis/'focused_cases'
    output.mkdir(exist_ok=False)
    load = lambda path: json.loads(path.read_text())
    summary = load(analysis/'analysis.json')
    package = root/'shared/processed/fp_dev250_v1'
    coco = load(package/'annotations_coco.json')
    anns = {g['id']: g for g in coco['annotations']}
    images = {i['id']: i for i in coco['images']}
    by_image = defaultdict(list)
    for g in anns.values():
        if not g['iscrowd']:
            by_image[g['image_id']].append(g)
    labels = {c['id']: c['name'] for c in coco['categories']}
    predictions, confusion = {}, {}
    for u in ('3000', '5000'):
        rows = load(root/f'runs/segmentation/segmentation-resume5000-ap-20261113-r01/update_{int(u):06d}/predictions.json')
        flat = [d for r in rows for d in r['detections']]
        predictions[u] = {i+1: d for i, d in enumerate(flat)}
        ledger = [json.loads(line) for line in (analysis/f'per_detection_{u}.jsonl').read_text().splitlines()]
        counts = Counter()
        cases = []
        for d in ledger:
            if d['category_id'] not in (3, 4) or d['score'] < .5 or d['error']['50'] != 'wrong_class_overlap':
                continue
            gs = by_image[d['image_id']]
            overlaps = mask_api.iou([predictions[u][d['detection_id']]['segmentation']], [g['segmentation'] for g in gs], [0]*len(gs))[0]
            g = gs[int(np.argmax(overlaps))]
            key = labels[d['category_id']]+' -> '+labels[g['category_id']]
            counts[key] += 1
            cases.append(dict(prediction=d, overlapping_gt_id=g['id'], overlapping_gt_category=g['category_id'], iou=float(max(overlaps))))
        confusion[u] = dict(counts=dict(counts), cases=cases)
    selected = []
    for name in ('skirt', 'outerwear', 'small'):
        selected.extend(dict(group=name, **r) for r in summary['groups'][name]['largest_same_iou_drops'][:2])
    tile_w, tile_h = 340, 320
    canvas = Image.new('RGB', (tile_w*4, tile_h*len(selected)), 'white')
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 13)
    except OSError:
        font = ImageFont.load_default()
    for row, case in enumerate(selected):
        gt = anns[case['annotation_id']]
        gt_mask = mask_api.decode(gt['segmentation']).astype(bool)
        im = Image.open(package/images[case['image_id']]['file_name']).convert('RGB')
        ys, xs = np.nonzero(gt_mask)
        cx, cy = (xs.min()+xs.max())/2, (ys.min()+ys.max())/2
        half_w, half_h = max(48, (xs.max()-xs.min()+1)*.7), max(48, (ys.max()-ys.min()+1)*.7)
        crop = (max(0,int(cx-half_w)), max(0,int(cy-half_h)), min(im.width,int(cx+half_w)+1), min(im.height,int(cy+half_h)+1))
        choices = [('GT '+labels[gt['category_id']], None, gt_mask),
                   ('3000 same class', case['old']['best_same'], None),
                   ('5000 same class', case['new']['best_same'], None),
                   ('5000 other class', case['new']['best_other'], None)]
        for col, (title, item, mask) in enumerate(choices):
            if item:
                u = '3000' if col == 1 else '5000'
                mask = (mask_api.decode(predictions[u][item['detection_id']]['segmentation']).astype(bool)
                        if item['detection_id'] else np.zeros_like(gt_mask))
            pixels = np.array(im).astype(float)
            pixels[mask] = .55*pixels[mask] + .45*np.array([20,220,40] if col==0 else [235,35,35])
            panel = Image.fromarray(pixels.astype('uint8')).crop(crop)
            panel.thumbnail((tile_w-12,tile_h-75))
            x,y=col*tile_w,row*tile_h
            canvas.paste(panel,(x+(tile_w-panel.width)//2,y+68))
            draw=ImageDraw.Draw(canvas)
            draw.text((x+6,y+4), title,fill='black',font=font)
            draw.text((x+6,y+22), f"{case['group']} img={case['image_id']}",fill='black',font=font)
            text=(f"IoU={item['iou']:.3f} score={item['score']:.4f} class={item['category_id']}" if item and item['score'] is not None
                  else f"ann={case['annotation_id']} area={case['area']}")
            draw.text((x+6,y+40),text,fill='black',font=font)
        case['diagnostic_crop_xyxy']=crop
    image_path=output/'paired_cases.png'
    canvas.save(image_path)
    result=dict(case_selection='two largest same-class best-IoU drops per skirt/outerwear/small; deliberately selected failures, not a random quality sample',
        high_score_wrong_class_overlaps=confusion, selected_cases=selected, inference_calls=0,
        image_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest(),
        source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output/'case_details.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'high_score_confusions':{u:x['counts'] for u,x in confusion.items()},'cases':len(selected),'inference_calls':0}))


if __name__=='__main__':
    main()
