"""Report independently eligible downloaded groups, not the unverified source universe."""
import argparse,sys
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools import mask as mask_api
sys.path.insert(0,str(Path(__file__).resolve().parent))
from expand_segmentation import SOURCE
from report_segmentation_expansion import summarize_records
from fashion_grounding.data.schema import read_json,write_json,sha256,require
def report(run):
 require(sha256(SOURCE)==read_json(run/"selection.json")["source_sha256"],"source changed")
 config=read_json(Path(__file__).resolve().parents[1]/"configs/source_category_mapping.json")
 mapping={x["source_id"]:x for x in config["fashionpedia"]}
 isolated={x["source_image_id"] for x in read_json(run/"duplicate_audit.json")["quarantine"]}
 fingerprints=[x for x in read_json(run/"new_fingerprints.json") if x["source_image"]["id"] not in isolated]
 coco=COCO();coco.dataset=read_json(SOURCE);coco.createIndex();images={};anns=[]
 for x in fingerprints:
  im=x["source_image"];iid=im["id"]+1
  images[iid]={"image_id":iid,"width":im["width"],"height":im["height"],"group_id":x["group_id"]}
  for a in coco.imgToAnns[im["id"]]:
   if mapping[a["category_id"]]["action"]!="map":continue
   anns.append({"image_id":iid,"category_id":mapping[a["category_id"]]["project_category_id"],"source_category_id":a["category_id"],"iscrowd":a["iscrowd"],"area":int(mask_api.area(coco.annToRLE(a)))})
 result=summarize_records(images,anns)
 result["scope"]="Downloaded oversampled pool after frozen-component isolation and one representative per new duplicate group; not all 39700 source candidates."
 write_json(run/"eligible_pool_report.json",result)
 print({k:result[k] for k in ("images","groups","instances","target_image_hits")})
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);a=p.parse_args();report(a.run)
