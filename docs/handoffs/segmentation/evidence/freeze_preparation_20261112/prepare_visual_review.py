"""One-off CPU review material; reuse published samples, never edit data."""
import hashlib
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from pycocotools import mask as mask_utils

ROOT = Path('/root/autodl-tmp/fashion-grounding')
OUT = ROOT / 'runs/segmentation/segmentation-freeze-preparation-20261112-r01/visual'
OUT.mkdir(exist_ok=False)
old_path = ROOT / 'runs/data_eval/20261112-large-data-regrouped/segmentation_preview/sample_manifest.json'
old = json.loads(old_path.read_text())['samples']
old_ids = {x['annotation_id'] for x in old}
items = []
for package in ('fp_train5000_v1', 'fp_dev250_v1'):
    base = ROOT / 'shared/processed' / package
    coco = json.loads((base / 'annotations_coco.json').read_text())
    images = {x['id']: x for x in coco['images']}
    items += [(base, images[a['image_id']], a) for a in coco['annotations']]
categories = {x['id']: x['name'] for x in coco['categories']}
report = {'seed': 20261112, 'reviewer': 'pending Codex visual inspection',
          'source': 'Fashionpedia', 'test_images_read': 0,
          'old_manifest_sha256': hashlib.sha256(old_path.read_bytes()).hexdigest(),
          'reused_samples': [], 'additional_samples': [], 'sheets': []}
for category in range(1, 9):
    eligible = [x for x in items if x[2]['category_id'] == category]
    reused = [x for x in eligible if x[2]['id'] in old_ids]
    fresh = random.Random(20261112 + category).sample(
        [x for x in eligible if x[2]['id'] not in old_ids], 20 - len(reused))
    def record(item):
        base, im, ann = item
        return dict(package=base.name, image_id=im['id'], annotation_id=ann['id'],
                    category_id=category, file_name=im['file_name'], image_sha256=im['sha256'])
    report['reused_samples'] += [record(x) for x in reused]
    report['additional_samples'] += [record(x) for x in fresh]
    sheet = Image.new('RGB', (1200, ((len(fresh) + 2) // 3) * 280), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, (base, im, ann) in enumerate(fresh):
        rgb = Image.open(base / im['file_name']).convert('RGB')
        mask = mask_utils.decode(ann['segmentation']).astype(bool)
        arr = np.asarray(rgb).copy()
        arr[mask] = (arr[mask] * .55 + np.array([255, 30, 200]) * .45).astype('uint8')
        ys, xs = np.where(mask)
        pad = max(20, int(max(xs.max()-xs.min(), ys.max()-ys.min()) * .25))
        box = (max(0, xs.min()-pad), max(0, ys.min()-pad),
               min(rgb.width, xs.max()+pad+1), min(rgb.height, ys.max()+pad+1))
        overlay = Image.fromarray(arr).crop(box)
        cx, cy = (i % 3)*400, (i // 3)*280
        sheet.paste(ImageOps.contain(rgb, (190, 235)), (cx, cy+35))
        sheet.paste(ImageOps.contain(overlay, (200, 235)), (cx+195, cy+35))
        draw.text((cx+3, cy+3), f'{categories[category]} ann={ann["id"]}', fill='black')
        draw.text((cx+3, cy+18), f'{base.name} img={im["id"]}', fill='black')
    name = f'class_{category}_additional.jpg'
    sheet.save(OUT / name, quality=93)
    report['sheets'].append(dict(file=name, count=len(fresh),
        sha256=hashlib.sha256((OUT/name).read_bytes()).hexdigest()))
(OUT / 'sample_manifest.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'reused': len(report['reused_samples']),
                  'additional': len(report['additional_samples']), 'out': str(OUT)}))
