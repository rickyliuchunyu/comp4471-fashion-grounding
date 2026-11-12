"""Build deterministic train/dev/held-out packages; conservative duplicate grouping."""
import argparse,json,random,shutil,sys,concurrent.futures
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps
from urllib.parse import urlsplit
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from fashion_grounding.data.schema import read_json,write_json,sha256,require,mask_box,validate_package
from fashion_grounding.data.convert import convert_image,COMMON

def phash(image):
 gray=np.asarray(image.convert("L").resize((32,32)),dtype=float)
 n=np.arange(32);basis=np.cos(np.pi*(2*n[None,:]+1)*n[:,None]/64)
 low=(basis@gray@basis.T)[:8,:8].flatten()[1:]
 return int(sum(int(b)<<i for i,b in enumerate(low>np.median(low))))

def specific_source_url(value):
 if not value:return None
 u=urlsplit(value)
 return value if u.scheme in ("http","https") and u.netloc and u.path.strip("/") else None

def group_images(items):
 """Conservatively group flags without claiming human-confirmed identity."""
 parent=list(range(len(items)))
 def find(i):
  while i!=parent[i]:parent[i]=parent[parent[i]];i=parent[i]
  return i
 def union(i,j):
  i,j=find(i),find(j)
  if i!=j:parent[max(i,j)]=min(i,j)
 exact={};urls={};pairs=[]
 for i,x in enumerate(items):
  for key,lookup in [(x["sha256"],exact),(specific_source_url(x.get("original_url")),urls)]:
   if not key:continue
   if key in lookup:union(i,lookup[key])
   else:lookup[key]=i
 h=np.array([x["phash"] for x in items],dtype=np.uint64)
 f=np.array([x["phash_flip"] for x in items],dtype=np.uint64)
 for i in range(len(items)):
  dist=np.minimum(np.bitwise_count(h[:i]^h[i]),np.bitwise_count(f[:i]^h[i]))
  for j in np.flatnonzero(dist<=6):
   j=int(j);union(i,j);pairs.append({"a":items[j]["key"],"b":items[i]["key"],"distance":int(dist[j]),"decision":"conservative_same_group_without_claiming_visual_identity"})
 groups=defaultdict(list)
 for i,x in enumerate(items):groups[find(i)].append(x["key"])
 assignment={}
 for values in groups.values():
  gid="fp-group-"+min(values)
  for key in values:assignment[key]=gid
 return assignment,pairs

def build_package(output,split,keys,inventory,sources,mapping,config,categories):
 require(not output.exists(),"output exists; choose version")
 output.mkdir(parents=True);(output/"images").mkdir();(output/"masks").mkdir()
 ims=[];cocoims=[];anns=[];cis=[];rawims=[];rawanns=[];issues=[];parts=0
 srcsplit="train" if split=="train" else "val"
 data=sources[srcsplit]["data"];by=sources[srcsplit]["by"]
 for n,key in enumerate(keys,1):
  inv=inventory[key];im=inv["source_image"];path=Path(inv["path"])
  try:converted,masks=convert_image(im,by[im["id"]],mapping,path)
  except (ValueError,KeyError,OSError,TypeError) as e:
   issues.append({"key":key,"error":str(e)});continue
  offset=0 if srcsplit=="train" else 10**9
  iid=im["id"]+1+offset
  for a in converted:a["id"]+=offset;a["image_id"]=iid
  filename="images/"+path.name;shutil.copyfile(path,output/filename)
  row={"image_id":iid,"file_name":filename,"sha256":inv["sha256"],"width":im["width"],"height":im["height"],"source_dataset":"fashionpedia","source_image_id":str(im["id"]),"source_split":"official_"+srcsplit+"2020","group_id":inv["group_id"],"split":split}
  ims.append(row);cocoims.append({**{k:v for k,v in row.items() if k!="image_id"},"id":iid,"license":im["license"],"supervised_category_ids":list(range(1,9))})
  anns.extend(converted);cs=[]
  for a in converted:
   if a["iscrowd"]:continue
   ref=f"masks/{a['id']}.json";write_json(output/ref,a["segmentation"])
   cs.append({"candidate_id":f"fp-{srcsplit}-gt-{a['id']}","category_id":a["category_id"],"score":None,"bbox_xyxy":mask_box(masks[a["id"]-offset]),"mask_reference":ref,"mask_sha256":sha256(output/ref),"decoder_query_id":None,"source_annotation_id":a["id"]})
  cis.append({"image_id":iid,"image_size_wh":[im["width"],im["height"]],"status":"ok" if cs else "no_candidates","candidates":cs})
  rawims.append(im);rawanns.extend(by[im["id"]]);parts+=sum(mapping[a["category_id"]]["action"]=="exclude_annotation" for a in by[im["id"]])
  if n%500==0:print("convert",split,n,len(keys),flush=True)
 write_json(output/"conversion_issues.json",issues)
 require(not issues,"image conversion failed, package not publishable")
 write_json(output/"mapping_config.json",config)
 write_json(output/"image_manifest.json",{**COMMON,"manifest_id":output.name,"mapping_version":config["mapping_version"],"images":ims})
 write_json(output/"annotations_coco.json",{**COMMON,"info":{"description":"Versioned eight-class "+split+" package","source_annotation_sha256":sources[srcsplit]["hash"]},"licenses":data["licenses"],"categories":categories,"images":cocoims,"annotations":anns})
 write_json(output/"candidates_gt.json",{**COMMON,"candidate_set_id":output.name+"-gt","candidate_source":"gt","image_manifest":{"path":"image_manifest.json","sha256":sha256(output/"image_manifest.json")},"provenance":{"producer_run_id":"wkq-large-data-v1","checkpoint_sha256":None,"decoder_config_sha256":None,"annotation_manifest_sha256":sha256(output/"annotations_coco.json")},"images":cis})
 write_json(output/"source_subset.json",{"info":data["info"],"licenses":data["licenses"],"categories":data["categories"],"attributes":data["attributes"],"images":rawims,"annotations":rawanns})
 for name in ("queries_draft.jsonl","queries_accepted.jsonl"):(output/name).write_text("")
 report={"images":len(ims),"instances":len(anns),"source_annotations":len(rawanns),"excluded_parts":parts,"crowd_instances":sum(a["iscrowd"]!=0 for a in anns),"instances_per_category":dict(Counter(a["category_id"] for a in anns)),"image_counts_per_category":{str(c):len({a["image_id"] for a in anns if a["category_id"]==c}) for c in range(1,9)},"area_bins":dict(Counter("small" if a["area"]<1024 else "medium" if a["area"]<9216 else "large" for a in anns)),"human_whole_instance_review":"not exhaustive; source masks preserved","queries_reviewed":False}
 write_json(output/"conversion_report.json",report)
 print("validate",split,flush=True);write_json(output/"validation_report.json",validate_package(output))
 return report

def main():
 p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);a=p.parse_args();raw=a.run/"raw"
 sel=read_json(raw/"selection.json")["selections"];repo=Path(__file__).resolve().parents[1]
 config=read_json(repo/"configs/source_category_mapping.json");mapping={r["source_id"]:r for r in config["fashionpedia"]}
 sources={};items=[]
 for split,selection in sel.items():
  source=read_json(selection["source_path"]);by=defaultdict(list)
  for ann in source["annotations"]:by[ann["image_id"]].append(ann)
  sources[split]={"data":source,"by":by,"hash":selection["source_sha256"]}
  for im in selection["images"]:
   items.append({"key":split+"-"+str(im["id"]),"source_image":im,"path":str(raw/split/Path(im["file_name"]).name),"source_split":split,"original_url":im.get("original_url")})
 # Include previously exposed training images in cross-split protection even outside the new random pool.
 known={x["key"] for x in items}
 shared=Path("/root/autodl-tmp/fashion-grounding/shared/processed")
 for name in ("fp_pilot20_v2","fp_grounded9_draft_v2"):
  root=shared/name;m=read_json(root/"image_manifest.json")
  for im in m["images"]:
   key="train-"+im["source_image_id"]
   if key in known:continue
   known.add(key);items.append({"key":key,"source_image":im,"path":str(root/im["file_name"]),"source_split":"train","prior_exposed_only":True,"original_url":None})
 def fingerprint(x):
  with Image.open(x["path"]) as im:
   im.load();require(im.size==(x["source_image"]["width"],x["source_image"]["height"]),"image size mismatch")
   x.update(sha256=sha256(x["path"]),phash=phash(im),phash_flip=phash(ImageOps.mirror(im)))
  return x
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:items=list(ex.map(fingerprint,items))
 print("fingerprints",len(items),flush=True)
 assignment,pairs=group_images(items);inventory={}
 for x in items:x["group_id"]=assignment[x["key"]];inventory[x["key"]]=x
 train_groups={x["group_id"] for x in items if x["source_split"]=="train"}
 quarantine=[x["key"] for x in items if x["source_split"]=="val" and x["group_id"] in train_groups]
 val_groups=defaultdict(list)
 for x in items:
  if x["source_split"]=="val" and x["key"] not in quarantine:val_groups[x["group_id"]].append(x["key"])
 groups=sorted(val_groups);random.Random(2026).shuffle(groups)
 splits={"train":[x["key"] for x in items if x["source_split"]=="train" and not x.get("prior_exposed_only")][:5000],"dev":[],"test":[]}
 used=set()
 for split,count in (("dev",250),("test",400)):
  for g in groups:
   if g not in used and len(splits[split])+len(val_groups[g])<=count:
    splits[split].extend(sorted(val_groups[g]));used.add(g)
   if len(splits[split])==count:break
  require(len(splits[split])==count,f"insufficient eligible {split}")
 require(len(splits["train"])==5000,"insufficient train")
 group_sets={s:{inventory[k]["group_id"] for k in keys} for s,keys in splits.items()}
 hash_sets={s:{inventory[k]["sha256"] for k in keys} for s,keys in splits.items()}
 for x,y in (("train","dev"),("train","test"),("dev","test")):
  require(not group_sets[x]&group_sets[y] and not hash_sets[x]&hash_sets[y],"split leakage")
 write_json(a.run/"fingerprints.json",items)
 write_json(a.run/"duplicate_audit.json",{"method":"SHA256 + exact resource-specific original URL (site homepages ignored) + 63-bit 32x32 DCT pHash Hamming<=6 including mirror; flags conservatively grouped, not asserted manually identical","near_duplicate_pairs":pairs,"cross_official_boundary_quarantined_validation":quarantine,"group_overlap":False,"hash_overlap":False,"limitations":["Not a proof that all crops, same-person or same-product relations are found","Source annotations may contain separated garment components; not merged"]})
 write_json(a.run/"split_plan.json",{"seed":2026,"splits":splits,"group_ids":{k:inventory[k]["group_id"] for v in splits.values() for k in v},"validation_unused":[x["key"] for x in items if x["source_split"]=="val" and x["key"] not in splits["dev"]+splits["test"]],"test_policy":"held-out images; no generated final-test language; no model selection","source_hashes":{s:d["hash"] for s,d in sources.items()},"freeze_state":"versioned_split_pending_team_acceptance"})
 categories=read_json(repo/"configs/categories.json")["categories"]
 reports={}
 for split in ("train","dev","test"):
  out=a.run/"packages"/{"train":"fp_train5000_v1","dev":"fp_dev250_v1","test":"fp_test400_holdout_v1"}[split]
  reports[split]=build_package(out,split,splits[split],inventory,sources,mapping,config,categories)
 write_json(a.run/"package_report.json",reports)
 print("BUILD_COMPLETE",flush=True)
if __name__=="__main__":main()
