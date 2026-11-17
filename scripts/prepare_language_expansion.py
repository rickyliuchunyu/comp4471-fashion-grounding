"""Select disjoint targeted language pools and render evidence before annotation."""
import sys,json,random,copy,shutil,os
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
from PIL import Image,ImageDraw,ImageOps
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from fashion_grounding.data.schema import read_json,write_json,sha256,decode_mask
from build_large_packages import build_package,specific_source_url
B=Path("/root/autodl-tmp/fashion-grounding");R=B/"runs/data_eval/20261117-language-targeted-expansion-r02"
P=B/"shared/processed";OLD=B/"runs/data_eval/20261112-large-data-regrouped"
def conflict(x,ys):
 for y in ys:
  if x["sha256"]==y["sha256"] or (specific_source_url(x.get("original_url")) and x.get("original_url")==y.get("original_url")):return True
  if min((int(x[a])^int(y[b])).bit_count() for a in ("phash","phash_flip") for b in ("phash","phash_flip"))<=6:return True
 return False
def choose(man,coco,count,excluded=set()):
 by=defaultdict(list)
 for a in coco["annotations"]:
  if not a["iscrowd"]:by[a["image_id"]].append(a)
 pools=defaultdict(list)
 for im in man["images"]:
  if im["group_id"] in excluded:continue
  aa=by[im["image_id"]];fine=Counter(a["source_category_id"] for a in aa)
  for a in aa:
   if a["area"]<128 or min(a["bbox"][2:])<8:continue
   s=a["source_category_id"]
   bucket="same_fine" if fine[s]>1 and a["source_category_id"] in (0,1,2,3,4,5,6,7,8,9,10,12,14,24,26) else "shoes" if a["category_id"]==6 else "bag" if a["category_id"]==7 else "accessory" if a["category_id"]==8 else None
   if bucket:pools[bucket].append((im,a))
 rng=random.Random(20261117)
 for pool in pools.values():rng.shuffle(pool)
 selected=[];used=set();groups=set();quota={"same_fine":int(count*.4),"shoes":int(count*.175),"bag":int(count*.2)}
 quota["accessory"]=count-sum(quota.values())
 for bucket,n in quota.items():
  for im,a in pools[bucket]:
   if im["image_id"] in used or im["group_id"] in groups:continue
   selected.append(dict(image_id=im["image_id"],target_id=a["id"],bucket=bucket))
   used.add(im["image_id"]);groups.add(im["group_id"])
   if sum(x["bucket"]==bucket for x in selected)>=n:break
 for bucket in pools:
  for im,a in pools[bucket]:
   if len(selected)>=count:break
   if im["image_id"] not in used and im["group_id"] not in groups:
    selected.append(dict(image_id=im["image_id"],target_id=a["id"],bucket=bucket));used.add(im["image_id"]);groups.add(im["group_id"])
 assert len(selected)==count,(len(selected),count)
 return selected
def render(root,selected,label):
 m=read_json(root/"image_manifest.json");ims={i["image_id"]:i for i in m["images"]}
 aa={a["id"]:a for a in read_json(root/"annotations_coco.json")["annotations"]}
 out=R/"sheets";out.mkdir(exist_ok=True)
 for start in range(0,len(selected),10):
  canvas=Image.new("RGB",(1200,1900),"white");d=ImageDraw.Draw(canvas)
  for j,s in enumerate(selected[start:start+10]):
   a=aa[s["target_id"]];im=ims[s["image_id"]];pic=Image.open(root/im["file_name"]).convert("RGB")
   x=(j%2)*600;y=(j//2)*380
   thumb=ImageOps.contain(pic,(380,330));canvas.paste(thumb,(x,y+35))
   scale=thumb.width/pic.width;bx,by,bw,bh=a["bbox"]
   d.rectangle((x+bx*scale,y+35+by*scale,x+(bx+bw)*scale,y+35+(by+bh)*scale),outline="lime",width=2)
   arr=np.asarray(pic).copy();mask=decode_mask(a["segmentation"],im["height"],im["width"])
   arr[mask]=(arr[mask]*.65+np.array([0,255,255])*.35).astype(np.uint8)
   crop=Image.fromarray(arr).crop((bx,by,bx+bw,by+bh))
   crop=ImageOps.contain(crop,(210,300));canvas.paste(crop,(x+385,y+45))
   n=start+j+1;s["sheet"]=f"{label}-{start//10+1:02d}.jpg";s["index"]=n
   d.text((x+4,y+3),f"{label}{n:03d} image={im['image_id']} target={a['id']}",fill="black")
   d.text((x+4,y+18),f"{a['source_category_name']} / {s['bucket']} / area={a['area']}",fill="black")
  canvas.save(out/f"{label}-{start//10+1:02d}.jpg",quality=90)
 write_json(R/(label+"-selection.json"),selected)
def main():
 R.mkdir(exist_ok=False,parents=True)
 train=P/"fp_train10000_targeted_v1";base=P/"fp_language1000_visual_draft_v3"
 excluded={i["group_id"] for i in read_json(base/"image_manifest.json")["images"]}
 ts=choose(read_json(train/"image_manifest.json"),read_json(train/"annotations_coco.json"),200,excluded)
 render(train,ts,"train")
 fps=read_json(OLD/"fingerprints.json")+read_json(B/"runs/data_eval/20261116-train10000-mirror4/new_fingerprints.json")
 bykey={x["key"]:x for x in fps}
 protected=[]
 for name in ("fp_train10000_targeted_v1","fp_dev250_v1","fp_test400_holdout_v1"):
  for im in read_json(P/name/"image_manifest.json")["images"]:
   key=("train-" if im["split"]=="train" else "val-")+im["source_image_id"];protected.append(bykey[key])
 unused=read_json(OLD/"split_plan.json")["validation_unused"]
 allowed=[];rejected=[];seen=[]
 for key in unused:
  x=bykey[key]
  if conflict(x,protected+seen):rejected.append(key)
  else:allowed.append(key);seen.append(x)
 write_json(R/"unused_val_screen.json",dict(eligible=allowed,rejected=rejected,protected_images=len(protected),method="SHA/resource URL/all four pHash orientations <=6; one representative per new component",limitations="Conservative image similarity, not exhaustive same-product identity"))
 rawsel=read_json(OLD/"raw/selection.json")["selections"]["val"]
 data=read_json(rawsel["source_path"]);by=defaultdict(list)
 for a in data["annotations"]:by[a["image_id"]].append(a)
 sources={"val":dict(data=data,by=by,hash=rawsel["source_sha256"])}
 config=read_json(Path("configs/source_category_mapping.json"));mapping={r["source_id"]:r for r in config["fashionpedia"]}
 cats=read_json(Path("configs/categories.json"))["categories"]
 # Metadata-only preselection; full source conversion only for selected dev images.
 fakeims=[];fakeanns=[]
 for key in allowed:
  x=bykey[key];im=x["source_image"];iid=10**9+im["id"]+1
  fakeims.append(dict(image_id=iid,group_id=x["group_id"]))
  for a in by[im["id"]]:
   r=mapping[a["category_id"]]
   if r["action"]!="map":continue
   z=dict(a,id=10**9+a["id"]+1,image_id=iid,source_category_id=a["category_id"],category_id=r["project_category_id"])
   fakeanns.append(z)
 ds=choose(dict(images=fakeims),dict(annotations=fakeanns),50)
 keys=["val-"+str(s["image_id"]-10**9-1) for s in ds]
 out=R/"dev50_source"
 build_package(out,"dev",keys,bykey,sources,mapping,config,cats)
 render(out,ds,"dev")
 print("prepared",len(ts),len(ds),len(allowed),flush=True)
if __name__=="__main__":main()
