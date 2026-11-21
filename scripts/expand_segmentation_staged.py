"""Append-only 15k/20k segmentation expansion; no models or language generation."""
import argparse, concurrent.futures, json, os, random, shutil, sys, time
from collections import Counter, defaultdict, deque
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
from pycocotools import mask as mu
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from fashion_grounding.data.schema import read_json, write_json, sha256, require, decode_mask, mask_box
from build_large_packages import phash, specific_source_url, build_package
from acquire_large_pool import download_members
from report_segmentation_expansion import summarize, summarize_records
B=Path("/root/autodl-tmp/fashion-grounding")
REPO=B/"workspaces/data_eval"
RUN=B/"runs/data_eval/20261118-segmentation-staged-v1"
SHARED=B/"shared/processed"
PARENT=SHARED/"fp_train10000_targeted_v1"
SOURCE=B/"runs/data_eval/20261112-grounded-queries/raw/instances_attributes_train2020.json"
PRIOR=B/"runs/data_eval/20261112-large-data-regrouped"
OLD_RUN=B/"runs/data_eval/20261116-train10000-mirror4"
SEED=2026
GOALS={"small_sock":200,"small_belt":300,"small_head_accessory":200,"small_watch":150,"small_shoe":700,"bag_all":400,"small_bag":50,"extreme_priority":300,"complex_boundary_source":300}
SMALL={22:"small_sock",19:"small_belt",15:"small_head_accessory",18:"small_watch",23:"small_shoe",24:"small_bag"}
PINS={"image_manifest.json":"9966383f85e1a4223a261f023eb80359b945815dff42ad4965c66c26225250d0","annotations_coco.json":"780b55fd3965606b41efeee62a7e708fcd45d3bd114b198d70f813f9f93f9f54","SHA256SUMS.json":"572b6d509f6cd759e697f1aae0280ded91776bc81a3126d2699b50c5d99462e3","mapping_config.json":"ce88a18ade3f6620eb83335b88da2e062847e033b7b75fd9a4266a480400583f"}
def sources():
 data=read_json(SOURCE); by=defaultdict(list)
 for a in data["annotations"]:by[a["image_id"]].append(a)
 return data,by
def raster_rle(a,im):
 seg=a["segmentation"];h,w=im["height"],im["width"]
 if isinstance(seg,list):return mu.merge(mu.frPyObjects(seg,h,w))
 if isinstance(seg["counts"],list):return mu.frPyObjects(seg,h,w)
 return {"size":seg["size"],"counts":seg["counts"].encode() if isinstance(seg["counts"],str) else seg["counts"]}
def hit_records(rows):
 hits=set()
 for a in rows:
  c=a.get("source_category_id",a["category_id"]);area=a["area"]
  if c in SMALL and area<=1024:hits.add(SMALL[c])
  if c==24:hits.add("bag_all")
  if c in SMALL and area<=256:hits.add("extreme_priority")
  if 308 in a.get("attribute_ids",[]):hits.add("complex_boundary_source")
 return hits
def totals(keys,stats):
 return {g:sum(g in stats[str(i)]["hits"] for i in keys) for g in GOALS}
def inventory():
 RUN.mkdir(parents=True,exist_ok=True)
 for f,h in PINS.items():require(sha256(PARENT/f)==h,"parent pin mismatch "+f)
 for package,pins in {"fp_dev250_v1":{"image_manifest.json":"c8c3b24d5a21f93a32060d2fdd9772102bf48b0d6504d60a3466444c7866ecdc","annotations_coco.json":"74837c386c4bbd2805f9976fcfd796931b2f6cfa72693b91d2f441bee473c635"},"fp_test400_holdout_v1":{"image_manifest.json":"ca8f5738c18cf800091e87eb36b86e80c9fe465c3de2adb4db7549f474419cf8"}}.items():
  for f,h in pins.items():require(sha256(SHARED/package/f)==h,"protection pin mismatch")
 data,by=sources();old=read_json(PARENT/"image_manifest.json")["images"];oldids={int(x["source_image_id"]) for x in old}
 stats={};excluded=Counter();errors=[]
 for n,im in enumerate(data["images"],1):
  iid=im["id"]
  if iid in oldids:continue
  rows=by[iid]
  if any(a["category_id"]==11 for a in rows):excluded["jumpsuit"]+=1;continue
  whole=[a for a in rows if a["category_id"]<27]
  if not whole:excluded["no_whole"]+=1;continue
  try:
   normalized=[]
   for a in whole:
    r=raster_rle(a,im);area=int(mu.area(r));box=mu.toBbox(r).tolist()
    require(area>0 and r["size"]==[im["height"],im["width"]],"empty/wrong dimension mask")
    normalized.append({**a,"area":area})
   stats[str(iid)]={"hits":sorted(hit_records(normalized)),"instances":len(whole),"annotation_areas":{str(a["id"]):a["area"] for a in normalized},"source_boundary_annotation_ids":[a["id"] for a in whole if 308 in a.get("attribute_ids",[])]}
  except Exception as e:errors.append({"source_image_id":iid,"error":str(e)});excluded["invalid_raster_metadata"]+=1
  if n%5000==0:print("inventory",n,flush=True)
 write_json(RUN/"metadata_stats.json",stats)
 ids=sorted(map(int,stats));random.Random(SEED).shuffle(ids)
 # Scan whole remaining source metadata, acquire a broad pool plus all priority rare cases.
 broad=ids[:14000]
 priority=[i for i in ids if set(stats[str(i)]["hits"])&{"small_sock","small_belt","small_head_accessory","small_watch","small_bag","complex_boundary_source","extreme_priority"}]
 pool=list(dict.fromkeys(broad+priority))
 cached={x["key"]:x for file in (PRIOR/"fingerprints.json",OLD_RUN/"new_fingerprints.json") for x in read_json(file)}
 report={"source_images":len(data["images"]),"parent_images":len(oldids),"metadata_raster_eligible":len(ids),"metadata_only_not_pixel_checked":True,"goals_metadata_availability":totals(ids,stats),"excluded":dict(excluded),"errors":errors,"acquisition_pool":len(pool),"pool_goals":totals(pool,stats),"cached_pool_fingerprints":sum("train-"+str(i) in cached for i in pool),"estimated_disk_gb":{"new_raw_and_packages_upper_estimate":30,"note":"rough estimate; copies may use extra disk; no old output deletion"},"complex_boundary_definition":"Official attribute 308 cutout only; source-supported label, not visual quality verification. Geometry proxy separate."}
 write_json(RUN/"feasibility_metadata.json",report);write_json(RUN/"acquisition_ids.json",pool);print(json.dumps(report),flush=True)
def acquire():
 data,_=sources();ids=set(read_json(RUN/"acquisition_ids.json"))
 cache={x["key"]:x for file in (PRIOR/"fingerprints.json",OLD_RUN/"new_fingerprints.json") for x in read_json(file)}
 raw=RUN/"raw";raw.mkdir(exist_ok=True);download=[]
 for im in data["images"]:
  if im["id"] not in ids:continue
  path=raw/Path(im["file_name"]).name;key="train-"+str(im["id"])
  if key in cache and Path(cache[key]["path"]).exists():
   if not path.exists():shutil.copyfile(cache[key]["path"],path)
   require(sha256(path)==cache[key]["sha256"],"cached copy mismatch")
  else:download.append(im)
 print("download_needed",len(download),flush=True)
 prov=download_members("https://s3.amazonaws.com/ifashionist-dataset/images/train2020.zip",download,raw,workers=32)
 write_json(RUN/"download_provenance.json",prov)
def incremental_groups(old,new):
 """Only new-old/new-new comparisons, inherited group relations included."""
 items=old+new;nold=len(old);parent=list(range(len(items)));edges=[]
 def find(i):
  while i!=parent[i]:parent[i]=parent[parent[i]];i=parent[i]
  return i
 def union(i,j):
  a,b=find(i),find(j)
  if a!=b:parent[max(a,b)]=min(a,b)
 group={};exact={};urls={}
 for i,x in enumerate(items):
  gid=x.get("group_id")
  if gid:
   if gid in group:union(i,group[gid])
   else:group[gid]=i
  for kind,key,lookup in [("sha256",x["sha256"],exact),("resource_url",specific_source_url(x.get("original_url")),urls)]:
   if key:
    if key in lookup:
     union(i,lookup[key])
     if i>=nold:edges.append({"a":items[lookup[key]]["key"],"b":x["key"],"kind":kind})
    else:lookup[key]=i
 h=np.array([x["phash"] for x in items],dtype=np.uint64);f=np.array([x["phash_flip"] for x in items],dtype=np.uint64)
 for i in range(nold,len(items)):
  d=np.minimum.reduce([np.bitwise_count(h[:i]^h[i]),np.bitwise_count(f[:i]^h[i]),np.bitwise_count(h[:i]^f[i]),np.bitwise_count(f[:i]^f[i])])
  for j in np.flatnonzero(d<=6):
   j=int(j);union(i,j);edges.append({"a":items[j]["key"],"b":items[i]["key"],"kind":"phash4","distance":int(d[j])})
 groups=defaultdict(list)
 for i,x in enumerate(items):groups[find(i)].append(x["key"])
 blocked=[];eligible=[];assignment={};oldkeys={x["key"] for x in old}
 for ks in groups.values():
  nk=sorted(k for k in ks if k not in oldkeys)
  if not nk:continue
  if set(ks)&oldkeys:blocked.extend({"key":k,"reason":"component_reaches_frozen_or_protected","component":ks} for k in nk)
  else:
   eligible.append(nk[0]);assignment[nk[0]]="fp-staged-20261118-"+nk[0]
   blocked.extend({"key":k,"reason":"new_component_representative_only","representative":nk[0]} for k in nk[1:])
 return eligible,assignment,edges,blocked
def select():
 data,_=sources();images={im["id"]:im for im in data["images"]};stats=read_json(RUN/"metadata_stats.json")
 cached={x["key"]:x for file in (PRIOR/"fingerprints.json",OLD_RUN/"new_fingerprints.json") for x in read_json(file)}
 # Preserve authoritative frozen group IDs; expand protection to all cached group members.
 frozen={}
 protected_packages=[PARENT,SHARED/"fp_dev250_v1",SHARED/"fp_test400_holdout_v1"]
 protection_inputs={}
 for p in protected_packages:
  protection_inputs[str(p/"image_manifest.json")]=sha256(p/"image_manifest.json")
  for im in read_json(p/"image_manifest.json")["images"]:
   k=("train-" if im["source_split"]=="official_train2020" else "val-")+im["source_image_id"]
   require(k in cached,"missing frozen fingerprint "+k);require(cached[k]["sha256"]==im["sha256"],"frozen fingerprint hash mismatch "+k);frozen[k]={**cached[k],"group_id":im["group_id"]}
 exposure_root=REPO/'docs/handoffs/data_eval'
 registries=sorted(exposure_root.glob('*_evidence/dev_exposure_registry.json'))
 require(exposure_root/'language_targeted_20261117_evidence/dev_exposure_registry.json' in registries,'missing original exposure registry')
 for exposure in registries:
  protection_inputs[str(exposure)]=sha256(exposure)
  for im in read_json(exposure)['images']:
   k='val-'+str(im['image_id']-10**9-1);require(k in cached,'missing exposure fingerprint');frozen[k]=cached[k]
 gids={x["group_id"] for x in frozen.values()}
 for k,x in cached.items():
  if x.get("group_id") in gids and k not in frozen:frozen[k]=x
 # Also reserve new development manifests (read metadata only).
 for p in sorted(SHARED.glob("fp_language_dev*/image_manifest.json")):
  protection_inputs[str(p)]=sha256(p)
  for im in read_json(p)["images"]:
   require(im["source_split"]=="official_val2020","dev source mismatch")
   k="val-"+im["source_image_id"];require(k in cached,"uncached additional dev");frozen.setdefault(k,cached[k])
 gids={x["group_id"] for x in frozen.values()}
 for k,x in cached.items():
  if x.get("group_id") in gids:frozen.setdefault(k,x)
 write_json(RUN/"protection_inputs.json",protection_inputs)
 write_json(RUN/"protected_fingerprints.json",list(frozen.values()))
 ids=read_json(RUN/"acquisition_ids.json");errors=[]
 def fp(i):
  im=images[i];k="train-"+str(i);p=RUN/"raw"/Path(im["file_name"]).name
  try:
   if k in cached:return {**cached[k],"source_image":im,"path":str(p),"group_id":None,"fingerprint_reused":True}
   with Image.open(p) as image:
    image.load();require(image.size==(im["width"],im["height"]),"size mismatch")
    return {"key":k,"source_image":im,"source_split":"train","path":str(p),"original_url":im.get("original_url"),"sha256":sha256(p),"phash":phash(image),"phash_flip":phash(ImageOps.mirror(image)),"group_id":None,"fingerprint_reused":False}
  except Exception as e:errors.append({"source_image_id":i,"error":str(e)});return None
 if (RUN/"pool_fingerprints.json").exists():new=read_json(RUN/"pool_fingerprints.json")
 else:
  with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:new=[x for x in ex.map(fp,ids) if x]
  write_json(RUN/"pool_fingerprints.json",new)
 # Any previously reserved component member remains blocked even when directly requested.
 direct=[x for x in new if x["key"] in frozen];new=[x for x in new if x["key"] not in frozen]
 eligible,assignment,edges,blocked=incremental_groups(list(frozen.values()),new)
 blocked.extend({"key":x["key"],"reason":"reserved_group_member"} for x in direct)
 inv={x["key"]:{**x,"group_id":assignment[x["key"]]} for x in new if x["key"] in assignment}
 eligible.sort();random.Random(SEED).shuffle(eligible)
 stages=[];remaining=eligible.copy()
 for phase in (1,2):
  require(len(remaining)>=5000,"fewer than 5000 eligible for phase "+str(phase))
  broad=remaining[:2500];rest=remaining[2500:];chosen=[];counts=Counter(g for k in broad for g in stats[k[6:]]["hits"])
  # Equivalent deficit-greedy selection grouped by hit signature; stable source-order ties.
  buckets=defaultdict(deque)
  for j,k in enumerate(rest):buckets[tuple(stats[k[6:]]["hits"])].append((j,k))
  for _ in range(2500):
   sig=max((sig for sig,q in buckets.items() if q),key=lambda sig:(sum(max(0,GOALS[g]-counts[g])/GOALS[g] for g in sig),-buckets[sig][0][0]))
   _,k=buckets[sig].popleft();chosen.append(k);counts.update(sig)
  keys=broad+chosen;used=set(keys);remaining=[k for k in remaining if k not in used]
  stages.append({"phase":phase,"keys":keys,"broad":broad,"targeted":chosen,"goals_actual":dict(counts),"goals_shortfall":{g:max(0,v-counts[g]) for g,v in GOALS.items()}})
 write_json(RUN/"selected.json",{"stages":stages,"seed":SEED})
 write_json(RUN/"eligible_inventory.json",inv)
 write_json(RUN/"duplicate_audit.json",{"old_old_recomputed":False,"orientations":4,"threshold":6,"protected_images":len(frozen),"edges":edges,"excluded":blocked,"invalid_images":errors,"eligible_independent":len(eligible),"limitations":"Conservative hash/URL/group isolation, not proof of all same-product/crop relationships"})
 write_json(RUN/"feasibility_checked.json",{"eligible_independent_pixel_fingerprinted":len(eligible),"eligible_goal_counts":totals([int(k[6:]) for k in eligible],stats),"pixel_invalid":errors,"excluded_components":len(blocked),"stages":[{k:v for k,v in s.items() if k not in ("keys","broad","targeted")} for s in stages],"pixel_masks_full_check":"new selected only during conversion; metadata RLE raster checked entire remaining source"})
 print("SELECTION_COMPLETE",len(eligible),[s["goals_actual"] for s in stages],flush=True)
def main():
 p=argparse.ArgumentParser();p.add_argument("step",choices=["inventory","acquire","select"]);a=p.parse_args();globals()[a.step]()
if __name__=="__main__":main()
