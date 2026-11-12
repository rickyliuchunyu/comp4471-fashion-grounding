"""Select a small training-only pool with same-category rivals for visual annotation."""
import argparse
from collections import defaultdict,Counter
from pathlib import Path
import random,sys,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from fashion_grounding.data.schema import read_json,write_json,sha256,decode_mask,require
from acquire_pilot import RemoteZip, IMAGES_URL, ANNOTATIONS_URL

def main():
 p=argparse.ArgumentParser();p.add_argument("--raw",type=Path,required=True);p.add_argument("--exclude-package",type=Path,required=True)
 args=p.parse_args();repo=Path(__file__).resolve().parents[1]
 source=args.raw/"instances_attributes_train2020.json"
 require(sha256(source)=="f6af94fb8712bd8170a49c857e842e1227e6806c4f33f32340db0c849133e006","source hash mismatch")
 data=read_json(source);mapping={r["source_id"]:r for r in read_json(repo/"configs/source_category_mapping.json")["fashionpedia"]}
 excluded={int(i["source_image_id"]) for i in read_json(args.exclude_package/"image_manifest.json")["images"]}
 grouped=defaultdict(list)
 for a in data["annotations"]:grouped[a["image_id"]].append(a)
 images=sorted(data["images"],key=lambda i:i["id"]);random.Random(20261112).shuffle(images)
 selected=[];used=set(excluded);counts=Counter()
 for cat in [7,1,4,5,2,3]:
  for im in images:
   if im["id"] in used:continue
   anns=grouped[im["id"]]
   if any(mapping[a["category_id"]]["action"]=="exclude_image" for a in anns):continue
   whole=[a for a in anns if mapping[a["category_id"]]["action"]=="map"]
   if not 3<=len(whole)<=24:continue
   rivals=[a for a in whole if mapping[a["category_id"]]["project_category_id"]==cat and not a["iscrowd"] and a["area"]>=0.015*im["width"]*im["height"]]
   if len(rivals)!=2 or len({a["category_id"] for a in rivals})!=1:continue
   centers=[a["bbox"][0]+a["bbox"][2]/2 for a in rivals]
   if abs(centers[0]-centers[1])<.15*im["width"]:continue
   try:
    for a in whole:decode_mask(a["segmentation"],im["height"],im["width"])
   except (ValueError,TypeError,KeyError):continue
   selected.append(im);used.add(im["id"]);counts[cat]+=1
   print("selected",im["id"],"category",cat,flush=True)
   if counts[cat]==2:break
 require(len(selected)>=8,"fewer than eight qualified images; review selection criteria")
 write_json(args.raw/"pilot_selection.json",{"seed":20261112,"count":len(selected),"source_split":"official_train2020","image_ids":sorted(i["id"] for i in selected),"source_annotation_sha256":sha256(source),"selection_policy":"two images per category 1..5 and 7; two non-crowd same-source-category rivals, each mask >=1.5% image, centers separated >=15% width; no jumpsuit; 3..24 whole instances; visual verification still required","selected_image_counts":dict(counts)})
 remote=RemoteZip(IMAGES_URL)
 with zipfile.ZipFile(remote) as archive:
  members={Path(x.filename).name:x for x in archive.infolist() if not x.is_dir()}
  folder=args.raw/"images";folder.mkdir(exist_ok=True)
  for n,im in enumerate(selected,1):
   name=Path(im["file_name"]).name;path=folder/name
   if not path.exists():path.write_bytes(archive.read(members[name]))
   print("downloaded",n,len(selected),im["id"],flush=True)
 write_json(args.raw/"download_provenance.json",{"annotation_url":ANNOTATIONS_URL,"annotation_sha256":sha256(source),"image_archive_url":IMAGES_URL,"archive_etag":remote.etag,"archive_bytes":remote.length,"range_bytes_transferred_this_run":remote.transferred,"images":[{"file_name":i["file_name"],"sha256":sha256(folder/Path(i["file_name"]).name)} for i in selected]})
if __name__=="__main__":main()
