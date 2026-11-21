"""Verify frozen parent bytes and independently recheck selected additions."""
import argparse,json,sys
from pathlib import Path
from collections import Counter
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from expand_segmentation import SHARED,FROZEN,PRIOR,frozen_snapshot
from build_large_packages import specific_source_url
from fashion_grounding.data.schema import read_json,write_json,sha256,require
def verify(run):
 before=read_json(run/"frozen_before.json");after=frozen_snapshot()
 require(before==after,"frozen packages changed")
 out=run/"fp_train10000_targeted_v1";added=run/"fp_added5000_targeted_v1";old=SHARED/FROZEN[0]
 require(read_json(out/"validation_report.json")["status"]=="passed","merged validation incomplete")
 require(read_json(added/"consumer_check_report.json")["status"]=="passed","added source check incomplete")
 for file,key,idkey in (("image_manifest.json","images","image_id"),("annotations_coco.json","images","id"),("annotations_coco.json","annotations","id"),("candidates_gt.json","images","image_id"),("source_subset.json","images","id"),("source_subset.json","annotations","id")):
  original=read_json(old/file)[key];combined={x[idkey]:x for x in read_json(out/file)[key]}
  require(all(combined[x[idkey]]==x for x in original),"old records differ")
 checked=0
 for path,value in before[FROZEN[0]]["files"].items():
  if path.startswith(("images/","masks/")):
   require(sha256(out/path)==value,"old asset bytes changed");checked+=1
 sel=set(read_json(run/"selected.json")["source_ids"])
 new=[x for x in read_json(run/"new_fingerprints.json") if x["source_image"]["id"] in sel]
 protected={}
 for name in FROZEN[:3]:
  for im in read_json(SHARED/name/"image_manifest.json")["images"]:
   key=("train-" if im["split"]=="train" else "val-")+im["source_image_id"];protected[key]=im["sha256"]
 original=[x for x in read_json(PRIOR/"fingerprints.json") if x["key"] in protected]
 require(all(x["sha256"]==protected[x["key"]] for x in original),"old fingerprint mismatch")
 items=original+new;h=np.array([x["phash"] for x in items],dtype=np.uint64);f=np.array([x["phash_flip"] for x in items],dtype=np.uint64)
 flags=[];hashes={x["sha256"] for x in original};urls={specific_source_url(x.get("original_url")) for x in original}-{None}
 for i in range(len(original),len(items)):
  x=items[i];u=specific_source_url(x.get("original_url"))
  require(x["sha256"] not in hashes and (not u or u not in urls),"new exact duplicate")
  hashes.add(x["sha256"])
  if u:urls.add(u)
  d=np.minimum.reduce([np.bitwise_count(h[:i]^h[i]),np.bitwise_count(f[:i]^h[i]),np.bitwise_count(h[:i]^f[i]),np.bitwise_count(f[:i]^f[i])])
  for j in np.flatnonzero(d<=6):flags.append({"a":items[int(j)]["key"],"b":x["key"],"distance":int(d[j])})
 write_json(run/"independent_duplicate_check.json",{"comparisons":"all four original/mirror orientations; new vs frozen and new vs new","threshold":6,"flags":flags})
 require(not flags,"new near-duplicate conflict")
 counts=Counter(x["sampling"] for x in read_json(run/"sampling_sidecar.json"))
 require(counts=={"broad":3500,"targeted":1500},"sampling counts")
 require(len(read_json(out/"image_manifest.json")["images"])==10000,"total images")
 source_old=read_json(old/"consumer_check_report.json");source_new=read_json(added/"consumer_check_report.json")
 report={"status":"passed","frozen_packages_byte_unchanged":list(FROZEN),"old_assets_identical_in_merged":checked,"old_json_records_equal":True,"new_independent_images":len(new),"new_duplicate_flags":0,"original_source_pixel_evidence":source_old["source_masks_pixel_identical"],"added_source_masks_pixel_identical":source_new["source_masks_pixel_identical"],"source_check_mode":"Reuse verified unchanged parent evidence; independently check all additions","model_smoke_performed":False,"training_performed":False}
 write_json(run/"preservation_report.json",report);write_json(out/"preservation_report.json",report)
 print(json.dumps(report,indent=2))
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);a=p.parse_args();verify(a.run)
