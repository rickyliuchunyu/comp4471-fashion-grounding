"""Decoded-mask size and class summaries for the append-only expansion."""
import argparse,json,sys
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from expand_segmentation import GOALS,hits
from fashion_grounding.data.schema import read_json,write_json
def summarize(root):
 m=read_json(root/"image_manifest.json");c=read_json(root/"annotations_coco.json")
 images={x["image_id"]:x for x in m["images"]};anns=c["annotations"]
 def table(field,ids):
  out={}
  for cid in ids:
   rows=[a for a in anns if a[field]==cid];iids={a["image_id"] for a in rows}
   small=[a for a in rows if a["area"]<=1024];tiny=[a for a in rows if a["area"]<=256]
   relative=[a["area"]/(images[a["image_id"]]["width"]*images[a["image_id"]]["height"]) for a in rows]
   out[str(cid)]={"instances":len(rows),"images":len(iids),"groups":len({images[i]["group_id"] for i in iids}),"small_instances_le1024":len(small),"small_images":len({a["image_id"] for a in small}),"extreme_instances_le256":len(tiny),"extreme_images":len({a["image_id"] for a in tiny}),"absolute_area_bins":dict(Counter("le256" if a["area"]<=256 else "257_1024" if a["area"]<=1024 else "1025_9216" if a["area"]<=9216 else "gt9216" for a in rows)),"relative_area_bins":dict(Counter("le0.001" if x<=.001 else "0.001_0.01" if x<=.01 else "0.01_0.1" if x<=.1 else "gt0.1" for x in relative)),"relative_area_quantiles_0_25_50_75_90_100":np.quantile(relative,[0,.25,.5,.75,.9,1]).tolist() if relative else None}
  return out
 by=defaultdict(list)
 for a in anns:by[a["image_id"]].append(a)
 counts=Counter(k for rows in by.values() for k in hits(rows))
 return {"images":len(images),"groups":len({im["group_id"] for im in images.values()}),"instances":len(anns),"crowd_instances":sum(bool(a["iscrowd"]) for a in anns),"small_instances_le1024":sum(a["area"]<=1024 for a in anns),"extreme_instances_le256":sum(a["area"]<=256 for a in anns),"exactly_1024_instances":sum(a["area"]==1024 for a in anns),"target_image_hits":dict(counts),"project_categories":table("category_id",range(1,9)),"source_categories":table("source_category_id",[i for i in range(27) if i!=11]),"area_basis":"decoded original-resolution source mask pixels; no image resizing"}
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);a=p.parse_args();r=a.run
 reports={"old":summarize(Path("/root/autodl-tmp/fashion-grounding/shared/processed/fp_train5000_v1")),"added":summarize(r/"fp_added5000_targeted_v1"),"combined":summarize(r/"fp_train10000_targeted_v1")}
 reports["soft_goals"]={k:{"goal":v,"actual_added_images":reports["added"]["target_image_hits"].get(k,0),"gap":max(0,v-reports["added"]["target_image_hits"].get(k,0))} for k,v in GOALS.items()}
 write_json(r/"distribution_report.json",reports)
 print(json.dumps({k:{f:d[f] for f in ("images","groups","instances","small_instances_le1024")} for k,d in reports.items() if k!="soft_goals"},indent=2));print(json.dumps(reports["soft_goals"],indent=2))
