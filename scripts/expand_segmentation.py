"""Append-only Fashionpedia segmentation expansion; frozen evaluation partitions."""
import argparse,concurrent.futures,json,random,shutil,sys
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from PIL import Image,ImageOps
from pycocotools.coco import COCO
from pycocotools import mask as mask_api
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from fashion_grounding.data.schema import read_json,write_json,sha256,require,validate_package
from build_large_packages import phash,group_images,build_package
from acquire_large_pool import download_members
ROOT=Path("/root/autodl-tmp/fashion-grounding")
SHARED=ROOT/"shared/processed"
SOURCE=ROOT/"runs/data_eval/20261112-grounded-queries/raw/instances_attributes_train2020.json"
PRIOR=ROOT/"runs/data_eval/20261112-large-data-regrouped"
FROZEN=("fp_train5000_v1","fp_dev250_v1","fp_test400_holdout_v1","fp_language1000_machine_v2","fp_language_dev_draft_v2")
GOALS={"small_sock":200,"small_belt":200,"small_head_accessory":200,"small_glasses":150,"small_shoe":500,"bag_all":500,"small_bag":80}
FINE={"small_sock":22,"small_belt":19,"small_head_accessory":15,"small_glasses":13,"small_shoe":23,"small_bag":24}
def hits(anns):
 out={k for k,c in FINE.items() if any(a.get("source_category_id",a["category_id"])==c and a["area"]<=1024 for a in anns)}
 if any(a.get("source_category_id",a["category_id"])==24 for a in anns):out.add("bag_all")
 return out
def choose_targeted(order,hit_map,count,initial=None):
 covered=Counter(initial or {});selected=[];remaining=[i for i in order if hit_map[i]]
 while remaining and len(selected)<count:
  deficits={k:max(0,v-covered[k]) for k,v in GOALS.items()}
  def score(i):
   hs=hit_map[i]
   return (sum(deficits[k]/GOALS[k] for k in hs),sum(1/(1+covered[k]) for k in hs))
  best=max(remaining,key=score)
  if not hit_map[best]:break
  selected.append(best);covered.update(hit_map[best]);remaining.remove(best)
 return selected
def frozen_snapshot():
 out={}
 for name in FROZEN:
  root=SHARED/name;expected=read_json(root/"SHA256SUMS.json")
  require(all(sha256(root/p)==v for p,v in expected.items()),"frozen checksum mismatch: "+name)
  out[name]={"checksums_file":sha256(root/"SHA256SUMS.json"),"files":expected}
 return out
def acquire(run):
 run.mkdir(parents=True,exist_ok=True)
 require(not (run/"selection.json").exists(),"selection exists; use resume command")
 write_json(run/"frozen_before.json",frozen_snapshot())
 source=read_json(SOURCE);by=defaultdict(list)
 for a in source["annotations"]:by[a["image_id"]].append(a)
 old={int(x["source_image_id"]) for x in read_json(SHARED/FROZEN[0]/"image_manifest.json")["images"]}
 eligible=[im for im in source["images"] if im["id"] not in old and not any(a["category_id"]==11 for a in by[im["id"]]) and any(a["category_id"]<27 for a in by[im["id"]])]
 order=sorted(im["id"] for im in eligible);random.Random(2026).shuffle(order)
 hm={i:hits([a for a in by[i] if a["category_id"]<27]) for i in order}
 broad=order[:4300];initial=Counter(k for i in broad[:3500] for k in hm[i])
 target=choose_targeted([i for i in order if i not in set(broad)],hm,2200,initial)
 wanted=broad+target;ims={im["id"]:im for im in eligible}
 selection={"seed":2026,"source_sha256":sha256(SOURCE),"eligible_metadata_images":len(eligible),"candidate_ids":wanted,"broad_order":broad,"targeted_order":target,"metadata_hits":{str(i):sorted(hm[i]) for i in wanted},"images":[ims[i] for i in wanted],"note":"Oversampled pool; quotas applied after conservative duplicate isolation. Source-area screening, decoded mask verification follows."}
 write_json(run/"selection.json",selection)
 print("pool",len(wanted),"broad_reserve",len(broad),"targeted_reserve",len(target),flush=True)
 resume(run)
def resume(run):
 sel=read_json(run/"selection.json");dest=run/"raw";dest.mkdir(exist_ok=True)
 oldraw=ROOT/"runs/data_eval/20261112-large-data/raw/train"
 for im in sel["images"]:
  name=Path(im["file_name"]).name
  if not (dest/name).exists() and (oldraw/name).exists():shutil.copyfile(oldraw/name,dest/name)
 provenance=download_members("https://s3.amazonaws.com/ifashionist-dataset/images/train2020.zip",sel["images"],dest,workers=12)
 write_json(run/"download_provenance.json",provenance);print("ACQUISITION_COMPLETE",flush=True)
def isolate_new_groups(candidate_ids,assignment,protected):
 blocked_groups={assignment[k] for k in protected};members=defaultdict(list)
 for key,group in assignment.items():members[group].append(key)
 quarantine=[];representatives={};available=[]
 # One representative per entirely new group; old groups/IDs never rewritten.
 for iid in candidate_ids:
  key="train-"+str(iid);g=assignment[key]
  if g in blocked_groups:
   quarantine.append({"source_image_id":iid,"reason":"group_touches_frozen_train_or_dev_test","component":g,"related_frozen_keys":[k for k in members[g] if k in protected]});continue
  if g in representatives:
   quarantine.append({"source_image_id":iid,"reason":"duplicate_with_new_pool_representative","component":g,"representative":representatives[g]});continue
  representatives[g]=iid;available.append(iid)
 return available,quarantine
def select(run):
 sel=read_json(run/"selection.json")
 protected={}
 for name in FROZEN[:3]:
  for im in read_json(SHARED/name/"image_manifest.json")["images"]:
   key=("train-" if im["split"]=="train" else "val-")+im["source_image_id"]
   protected[key]={"frozen_package":name,"frozen_group_id":im["group_id"],"frozen_sha256":im["sha256"]}
 prior={x["key"]:x for x in read_json(PRIOR/"fingerprints.json")}
 items=[]
 for key,info in protected.items():
  x=dict(prior[key]);require(x["sha256"]==info["frozen_sha256"],"stale protection fingerprint");x.update(info);items.append(x)
 def fingerprint(im):
  path=run/"raw"/Path(im["file_name"]).name
  with Image.open(path) as pic:
   pic.load();require(pic.size==(im["width"],im["height"]),"size mismatch")
   return {"key":"train-"+str(im["id"]),"source_image":im,"path":str(path),"source_split":"train","original_url":im.get("original_url"),"sha256":sha256(path),"phash":phash(pic),"phash_flip":phash(ImageOps.mirror(pic))}
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:new=list(ex.map(fingerprint,sel["images"]))
 require(sha256(SOURCE)==sel["source_sha256"],"source annotations changed")
 source=read_json(SOURCE);coco=COCO();coco.dataset=source;coco.createIndex()
 verified_hits={};area_changes=0;issues=[]
 for x in new:
  iid=x["source_image"]["id"];rows=[]
  try:
   for ann in coco.imgToAnns[iid]:
    if ann["category_id"]>=27:continue
    require(ann["category_id"]!=11,"jumpsuit entered pool")
    area=int(mask_api.area(coco.annToRLE(ann)))
    require(area>0,"empty mask")
    area_changes+=area!=ann["area"];rows.append({**ann,"area":area})
   require(bool(rows),"no whole masks")
   verified_hits[iid]=hits(rows)
  except (ValueError,KeyError,TypeError,IndexError) as exc:issues.append({"source_image_id":iid,"error":str(exc),"decision":"quarantine_entire_image"})
 write_json(run/"pool_mask_check.json",{"checked_images":len(new),"valid_images":len(verified_hits),"source_area_differs_from_raster_count":area_changes,"issues":issues,"area_basis":"pycocotools RLE area at original resolution"})
 new=[x for x in new if x["source_image"]["id"] in verified_hits]
 items+=new;assignment,pairs=group_images(items,all_orientations=True)
 available,quarantine=isolate_new_groups([i for i in sel["candidate_ids"] if i in verified_hits],assignment,protected)
 avail=set(available);broad=[i for i in sel["broad_order"] if i in avail][:3500]
 require(len(broad)==3500,"insufficient broad independent groups")
 hm=verified_hits
 initial=Counter(k for i in broad for k in hm[i])
 candidates=[i for i in available if i not in set(broad)]
 target=choose_targeted(candidates,hm,1500,initial)
 require(len(target)==1500,"insufficient targeted groups")
 selected=broad+target;inventory={x["key"]:x for x in new}
 for x in new:x["group_id"]="fp-expansion-"+assignment[x["key"]]
 write_json(run/"new_fingerprints.json",new)
 write_json(run/"duplicate_audit.json",{"method":"SHA256, resource-specific URL, all four original/mirror 63-bit pHash comparisons distance <=6; conservative connected components; no visual identity claims","protected_images":len(protected),"pool_images":len(new),"independent_eligible_images":len(available),"independent_eligible_groups":len(available),"near_pairs":pairs,"quarantine":quarantine,"frozen_records_reassigned":False})
 sidecar=[]
 for i in selected:
  im=inventory["train-"+str(i)]
  sidecar.append({"image_id":i+1,"source_image_id":i,"sampling":"broad" if i in set(broad) else "targeted","target_reasons":sorted(hm[i]),"group_id":im["group_id"],"original_url":im.get("original_url"),"sha256":im["sha256"],"difficulty_or_shape":"unknown unless explicitly present in preserved source attributes"})
 write_json(run/"sampling_sidecar.json",sidecar)
 write_json(run/"selected.json",{"broad":broad,"targeted":target,"source_ids":selected,"broad_hits":dict(initial),"added_hits_decoded_area":dict(Counter(k for i in selected for k in hm[i])),"decoded_area_soft_goals":GOALS})
 print("SELECT_COMPLETE",len(selected),"quarantined",len(quarantine),flush=True)
def build(run):
 chosen=read_json(run/"selected.json");source=read_json(SOURCE);by=defaultdict(list)
 for a in source["annotations"]:by[a["image_id"]].append(a)
 repo=Path(__file__).resolve().parents[1];config=read_json(repo/"configs/source_category_mapping.json");mapping={x["source_id"]:x for x in config["fashionpedia"]}
 sources={"train":{"data":source,"by":by,"hash":sha256(SOURCE)}}
 inventory={x["key"]:x for x in read_json(run/"new_fingerprints.json")}
 added=run/"fp_added5000_targeted_v1"
 build_package(added,"train",["train-"+str(i) for i in chosen["source_ids"]],inventory,sources,mapping,config,read_json(repo/"configs/categories.json")["categories"])
 print("ADDED_COMPLETE",flush=True)
def merge(run):
 old=SHARED/FROZEN[0];added=run/"fp_added5000_targeted_v1";out=run/"fp_train10000_targeted_v1"
 require(not out.exists(),"merged output exists")
 out.mkdir()
 for folder in ("images","masks"):
  shutil.copytree(old/folder,out/folder)
  for p in (added/folder).iterdir():
   require(not (out/folder/p.name).exists(),"file collision");shutil.copyfile(p,out/folder/p.name)
 for file,key in (("image_manifest.json","images"),("annotations_coco.json","images"),("candidates_gt.json","images"),("source_subset.json","images")):
  obj=read_json(old/file);extra=read_json(added/file)
  obj[key]+=extra[key]
  if file in ("annotations_coco.json","source_subset.json"):obj["annotations"]+=extra["annotations"]
  if file=="image_manifest.json":obj["manifest_id"]=out.name
  if file=="candidates_gt.json":obj["candidate_set_id"]=out.name+"-gt"
  write_json(out/file,obj)
 cs=read_json(out/"candidates_gt.json");cs["image_manifest"]["sha256"]=sha256(out/"image_manifest.json");cs["provenance"]["annotation_manifest_sha256"]=sha256(out/"annotations_coco.json");cs["provenance"]["producer_run_id"]="wkq-train10000-targeted-20261116";write_json(out/"candidates_gt.json",cs)
 shutil.copyfile(old/"mapping_config.json",out/"mapping_config.json")
 for name in ("queries_draft.jsonl","queries_accepted.jsonl"):(out/name).write_text("")
 shutil.copyfile(run/"sampling_sidecar.json",out/"sampling_sidecar.json")
 write_json(out/"old_subset_membership.json",{"parent_package":str(old),"image_ids":[x["image_id"] for x in read_json(old/"image_manifest.json")["images"]],"old_image_id_to_new_image_id":"identity","old_annotation_id_to_new_annotation_id":"identity","new_source_to_project_id":"source_id + 1"})
 # Object equality preserves all old fields, in addition to unchanged image/mask bytes.
 for file,key,idkey in (("image_manifest.json","images","image_id"),("annotations_coco.json","images","id"),("annotations_coco.json","annotations","id"),("candidates_gt.json","images","image_id"),("source_subset.json","images","id"),("source_subset.json","annotations","id")):
  original=read_json(old/file)[key];combined={x[idkey]:x for x in read_json(out/file)[key]}
  require(all(combined[x[idkey]]==x for x in original),"old records changed: "+file)
 write_json(out/"validation_report.json",validate_package(out))
 print("MERGE_COMPLETE",flush=True)
def main():
 p=argparse.ArgumentParser();p.add_argument("step",choices=("acquire","resume","select","build","merge"));p.add_argument("--run",type=Path,required=True);a=p.parse_args();globals()[a.step](a.run)
if __name__=="__main__":main()
