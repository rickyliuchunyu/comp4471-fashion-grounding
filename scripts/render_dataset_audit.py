"""Deterministic per-class mask/query contact sheets; no model execution."""
import argparse,json,random,sys,textwrap
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from fashion_grounding.data.schema import read_json,decode_mask,write_json

def render(package,output,per_class=20):
 output.mkdir(parents=True,exist_ok=True)
 manifest=read_json(package/"image_manifest.json");ims={im["image_id"]:im for im in manifest["images"]}
 coco=read_json(package/"annotations_coco.json");anns={a["id"]:a for a in coco["annotations"]}
 query={}
 for line in (package/"queries_draft.jsonl").read_text().splitlines():
  q=json.loads(line)
  if q["language"]=="en":query.setdefault(q["target_instance_id"],q)
 groups=defaultdict(list)
 for a in anns.values():
  if a["iscrowd"]:continue
  if query and a["id"] not in query:continue
  groups[a["category_id"]].append(a)
 selected=[]
 for cat,group in sorted(groups.items()):
  group.sort(key=lambda a:a["id"]);random.Random(2026+cat).shuffle(group);group=group[:per_class]
  sheet=Image.new("RGB",(1600,300*((len(group)+3)//4)),"white");draw=ImageDraw.Draw(sheet)
  for i,a in enumerate(group):
   x=(i%4)*400;y=(i//4)*300;im=ims[a["image_id"]]
   rgb=Image.open(package/im["file_name"]).convert("RGB");full=rgb.copy();full.thumbnail((185,230));sheet.paste(full,(x,y+20))
   m=decode_mask(a["segmentation"],im["height"],im["width"])
   pixels=np.asarray(rgb).copy();pixels[~m]=235
   bx,by,bw,bh=a["bbox"];crop=Image.fromarray(pixels).crop((bx,by,bx+bw,by+bh));crop.thumbnail((200,230));sheet.paste(crop,(x+190,y+20))
   q=query.get(a["id"]);text=q["query"] if q else a["source_category_name"]
   draw.text((x+5,y+2),f"img {a['image_id']} / ann {a['id']}",fill="black")
   draw.multiline_text((x+5,y+252),"\n".join(textwrap.wrap(text,55)[:3]),fill="black",spacing=2)
   selected.append({"image_id":a["image_id"],"annotation_id":a["id"],"category_id":cat,"query_id":q["query_id"] if q else None})
  sheet.save(output/f"class_{cat}.jpg",quality=92)
 write_json(output/"sample_manifest.json",{"seed":2026,"per_class":per_class,"samples":selected,"review_status":"rendered_not_automatically_reviewed"})
 return len(selected)
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--package",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--per-class",type=int,default=20);a=p.parse_args()
 print(render(a.package,a.output,a.per_class))
