"""Render deliberate saved-mask examples. Diagnostic GT crops never feed inference."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from pycocotools import mask as mask_api
from analyze_saved_segmentation_predictions import digest,write


def select_cases(analysis):
    s=json.loads((analysis/'small_errors.json').read_text());old_u,new_u=s['updates']
    old=json.loads((analysis/f'small_gt_{old_u}.json').read_text())
    new={r['annotation_id']:r for r in json.loads((analysis/f'small_gt_{new_u}.json').read_text())}
    cases=list(s['selected_cases']); used={r['annotation_id'] for r in cases}
    conflicts=sorted([r for r in old if r['annotation_id'] not in used and r['highest_score_overlap'] and r['highest_score_overlap']['category_id']!=r['category_id']],key=lambda r:r['annotation_id'])
    if conflicts:
        r=conflicts[0];cases.append(dict(kind='class_score_conflict',annotation_id=r['annotation_id'],image_id=r['image_id'],old=r,new=new[r['annotation_id']]))
    return s['updates'],cases


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--analysis',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root,analysis,out=Path(a.root),Path(a.analysis),Path(a.output);out.mkdir(exist_ok=False)
    cfg=json.loads((analysis/'resolved_config.json').read_text());package=root/cfg['package'];coco=json.loads((package/'annotations_coco.json').read_text());anns={g['id']:g for g in coco['annotations']};ims={i['id']:i for i in coco['images']}
    updates,cases=select_cases(analysis);predictions={}
    for item in cfg['inputs']:
        path=root/item['directory']/'predictions.json'
        expected=json.loads((analysis/'run_metadata.json').read_text())['inputs'][str(item['update'])]['predictions_sha256'];assert digest(path)==expected
        flat=[d for row in json.loads(path.read_text()) for d in row['detections']];predictions[str(item['update'])]={i+1:d for i,d in enumerate(flat)}
    w,h=340,290;canvas=Image.new('RGB',(w*5,h*len(cases)+52),'white')
    font_path='/System/Library/Fonts/Supplemental/Arial.ttf'
    font=ImageFont.truetype(font_path,14);titlefont=ImageFont.truetype(font_path,20)
    ImageDraw.Draw(canvas).text((10,9),'Saved dev predictions: selected failure/change examples (GT green, predictions red)',font=titlefont,fill='black')
    for row,c in enumerate(cases):
        g=anns[c['annotation_id']];im=ims[c['image_id']];imagepath=package/im['file_name'];assert digest(imagepath)==im['sha256']
        original=Image.open(imagepath).convert('RGB');gm=mask_api.decode(g['segmentation']).astype(bool);ys,xs=np.nonzero(gm)
        cx,cy=(xs.min()+xs.max())/2,(ys.min()+ys.max())/2;hw=max(32,(xs.max()-xs.min()+1)*.85);hh=max(32,(ys.max()-ys.min()+1)*.85)
        crop=(max(0,int(cx-hw)),max(0,int(cy-hh)),min(original.width,int(cx+hw)+1),min(original.height,int(cy+hh)+1))
        choices=[('Original crop',None,None,None),('GT '+g['source_category_name'],gm,None,None),
                 (updates[0]+' same class',None,c['old']['best_same'],updates[0]),
                 (updates[1]+' same class',None,c['new']['best_same'],updates[1]),
                 (updates[1]+' other class',None,c['new']['best_other'],updates[1])]
        for col,(title,mask,item,u) in enumerate(choices):
            pixels=np.asarray(original).astype(float)
            if item:mask=mask_api.decode(predictions[u][item['detection_id']]['segmentation']).astype(bool) if item['detection_id'] else np.zeros_like(gm)
            if mask is not None:pixels[mask]=pixels[mask]*.55+.45*np.array([15,225,70] if col==1 else [235,40,35])
            panel=Image.fromarray(pixels.astype('uint8')).crop(crop)
            zoom=min((w-12)/panel.width,(h-86)/panel.height)
            panel=panel.resize((round(panel.width*zoom),round(panel.height*zoom)),Image.Resampling.NEAREST)
            x,y=col*w,52+row*h;canvas.paste(panel,(x+(w-panel.width)//2,y+80));d=ImageDraw.Draw(canvas)
            d.text((x+5,y+3),title[:33],font=font,fill='black');d.text((x+5,y+23),c['kind']+' | GT '+str(c['annotation_id']),font=font,fill='black')
            line=(f"IoU {item['iou']:.3f} score {item['score']:.4f} class {item['category_id']}" if item and item['score'] is not None else f"area {g['area']} px | image {c['image_id']}")
            d.text((x+5,y+43),line,font=font,fill='black')
        c['crop_xyxy']=crop;c['source_image_sha256']=im['sha256'];c['source_image_path']=str(imagepath)
    image=out/'small_object_cases.png';canvas.save(image)
    write(out/'case_details.json',dict(selected_cases=cases,selection_rule='Saved small_errors deterministic cases plus first annotation-ID class-score conflict. Deliberate failures/changes, not random or representative.',image_sha256=digest(image),source_script_sha256=digest(__file__),inference_calls=0,optimizer_updates=0))
    print(image)
if __name__=='__main__':main()
