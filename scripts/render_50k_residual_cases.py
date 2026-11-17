"""Render existing masks as diagnostic figures; no inference or mask repair."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pycocotools import mask as mask_api
from analyze_resolution_trial import inner_boundary
from analyze_saved_segmentation_predictions import digest, write
from analyze_50k_residual_errors import normalized_rle


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--analysis',required=True)
    a=p.parse_args();root=Path(a.root);analysis=Path(a.analysis)
    cfg=json.loads((analysis/'resolved_config.json').read_text());cases=json.loads((analysis/'selected_cases.json').read_text())
    coco=json.loads((root/cfg['annotations']).read_text());anns={g['id']:g for g in coco['annotations']}
    manifest=json.loads((root/'shared/processed/fp_dev250_v1/image_manifest.json').read_text());ims={im['image_id']:im for im in manifest['images']}
    cache_root=root/cfg['cache'];cache=json.loads((cache_root/'cache.json').read_text());refs={r['image_id']:r for r in cache['records']}
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
    small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    artifacts=[];ledger=[]
    for page in range((len(cases)+3)//4):
        canvas=Image.new('RGB',(1080,1080),'white');draw=ImageDraw.Draw(canvas)
        draw.text((15,8),'50k saved-mask failure examples (selected illustrations, not a random sample)',fill='black',font=font)
        draw.text((15,33),'Context crop | GT (green) | Saved prediction (magenta); raw cache shown for filtered/missing cases',fill='black',font=small)
        for i,c in enumerate(cases[page*4:(page+1)*4]):
            y=65+i*250;g=anns[c['annotation_id']];im=ims[c['image_id']]
            record=json.loads((cache_root/refs[c['image_id']]['path']).read_text());cs={r['candidate_id']:r for r in record['candidates']}
            candidate_id=c['matched50_candidate_id'] if c['kind']=='boundary' else (
                c['raw']['best_other']['candidate_id'] if c['kind']=='wrong_class_only' else c['raw']['best_same']['candidate_id'])
            candidate=cs[candidate_id]
            gm=mask_api.decode(normalized_rle(g['segmentation'])).astype(bool)
            pm=mask_api.decode(normalized_rle(json.loads((cache_root/candidate['mask_reference']).read_text()))).astype(bool)
            with Image.open(root/'shared/processed/fp_dev250_v1'/im['file_name']) as f:rgb=np.asarray(f.convert('RGB')).copy()
            x0,y0,w,h=g['bbox'];cx,cy=x0+w/2,y0+h/2;cw,ch=max(96,w*2),max(96,h*2)
            box=(max(0,int(cx-cw/2)),max(0,int(cy-ch/2)),min(im['width'],int(cx+cw/2)),min(im['height'],int(cy+ch/2)))
            intersect=int((gm&pm).sum());union=int((gm|pm).sum());iou=intersect/union if union else 0.
            draw.text((15,y),f"{c['source_category']} | {c['kind']} | GT {g['id']} | area {g['area']} | {c['residual_cause']}",fill='black',font=small)
            draw.text((15,y+20),f"pred class {candidate['category_id']} / score {candidate['score']:.6f} / mask IoU {iou:.3f} / query {candidate['decoder_query_id']}",fill='black',font=small)
            for col,mask,color in ((0,None,None),(1,gm,np.array([0,220,80])),(2,pm,np.array([235,0,160]))):
                pixels=rgb.copy()
                if mask is not None:
                    pixels[mask]=(.65*pixels[mask]+.35*color).astype('uint8')
                    pixels[inner_boundary(mask)]=color
                tile=Image.fromarray(pixels).crop(box);tile.thumbnail((340,195),Image.Resampling.NEAREST)
                canvas.paste(tile,(15+col*355+(340-tile.width)//2,y+48))
            ledger.append(dict(annotation_id=g['id'],image_id=c['image_id'],kind=c['kind'],crop_box=box,
                candidate_id=candidate_id,score=candidate['score'],category_id=candidate['category_id'],mask_iou=iou,
                artifact=f'failure_cases_{page+1}.png'))
        target=analysis/f'failure_cases_{page+1}.png';assert not target.exists();canvas.save(target)
        artifacts.append(dict(path=str(target),sha256=digest(target)))
    assert 'torch' not in sys.modules
    write(analysis/'visual_receipt.json',dict(status='completed',source_sha256=digest(__file__),artifacts=artifacts,cases=ledger,
        inference_calls=0,source_masks_changed=False,selection='Deterministic examples from all-GT statistics; no labels corrected or candidate payloads modified.'))
    print(json.dumps(artifacts))


if __name__=='__main__':main()
