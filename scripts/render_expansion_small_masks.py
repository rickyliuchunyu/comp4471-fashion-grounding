"""Inspect original-resolution small masks without changing training images."""
import argparse,random,sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image,ImageDraw,ImageOps
sys.path.insert(0,str(Path(__file__).resolve().parent))
from expand_segmentation import FINE
from fashion_grounding.data.schema import read_json,write_json,decode_mask
def render(root,out):
 out.mkdir(parents=True,exist_ok=False)
 ims={x["image_id"]:x for x in read_json(root/"image_manifest.json")["images"]}
 anns=read_json(root/"annotations_coco.json")["annotations"];samples=[]
 for n,(name,cid) in enumerate(FINE.items()):
  pool=[a for a in anns if a["source_category_id"]==cid and a["area"]<=1024]
  random.Random(2026+n).shuffle(pool)
  tiny=[a for a in pool if a["area"]<=256][:4]
  rest=[a for a in pool if a not in tiny];chosen=(tiny+rest)[:8]
  sheet=Image.new("RGB",(1600,680),"white");draw=ImageDraw.Draw(sheet)
  for j,a in enumerate(chosen):
   im=ims[a["image_id"]];rgb=Image.open(root/im["file_name"]).convert("RGB")
   x=(j%4)*400;y=(j//4)*340;full=rgb.copy();full.thumbnail((190,280));sheet.paste(full,(x,y+30))
   mask=decode_mask(a["segmentation"],im["height"],im["width"])
   overlay=np.asarray(rgb).copy();overlay[mask]=(overlay[mask]*.55+np.array([0,220,255])*.45).astype(np.uint8)
   bx,by,w,h=a["bbox"];box=(max(0,bx-8),max(0,by-8),min(im["width"],bx+w+8),min(im["height"],by+h+8))
   crop=ImageOps.contain(Image.fromarray(overlay).crop(box),(200,280),method=Image.Resampling.NEAREST)
   sheet.paste(crop,(x+195,y+30))
   draw.text((x+3,y+3),f"im {a['image_id']} / ann {a['id']}",fill="black")
   draw.text((x+3,y+312),f"{name}; area={a['area']}; zoom only",fill="black")
   samples.append({"image_id":a["image_id"],"annotation_id":a["id"],"source_category_id":cid,"area":a["area"],"sample_group":name})
  sheet.save(out/(name+".jpg"),quality=94)
 write_json(out/"sample_manifest.json",{"samples":samples,"status":"rendered_pending_visual_inspection","note":"Nearest-neighbor display zoom only; source data unchanged"})
 print(len(samples))
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("package",type=Path);p.add_argument("--output",type=Path,required=True);a=p.parse_args();render(a.package,a.output)
