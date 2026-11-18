"""Build and publish staged segmentation expansions, reusing pinned parent audits."""
import argparse, csv, json, platform, random, shutil, subprocess, sys
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageOps
from expand_segmentation_staged import *
from fashion_grounding.data.schema import validate_package
def verify_membership(parent,delta,combined):
 for file,field,key in [("image_manifest.json","images","image_id"),("annotations_coco.json","images","id"),("annotations_coco.json","annotations","id"),("candidates_gt.json","images","image_id"),("source_subset.json","images","id"),("source_subset.json","annotations","id")]:
  old=read_json(parent/file)[field];new=read_json(delta/file)[field];allrows=read_json(combined/file)[field]
  oi={r[key]:r for r in old};ni={r[key]:r for r in new};ci={r[key]:r for r in allrows}
  require(len(oi)==len(old) and len(ni)==len(new) and len(ci)==len(allrows),"duplicate ID")
  require(not oi.keys()&ni.keys() and ci=={**oi,**ni},"parent/delta identity failure "+file+field)
 return True
def sidecar_and_source_check(delta,stage,source_by,source_images):
 coco=read_json(delta/"annotations_coco.json");manifest=read_json(delta/"image_manifest.json");images={im["image_id"]:im for im in manifest["images"]}
 selected_source_ids={int(i["source_image_id"]) for i in manifest["images"]}
 expected={a["id"] for sid in selected_source_ids for a in source_by[sid] if a["category_id"]<27 and a["category_id"]!=11}
 require({a["source_annotation_id"] for a in coco["annotations"]}==expected,"source annotation coverage")
 mapping={r["source_id"]:r for r in read_json(delta/"mapping_config.json")["fashionpedia"]}
 require(all(a["category_id"]==mapping[a["source_category_id"]]["project_category_id"] for a in coco["annotations"]),"source category mapping")
 by=defaultdict(list);evidence=[];checked=0;geometry=[];stats=read_json(RUN/"metadata_stats.json")
 for a in coco["annotations"]:
  by[a["image_id"]].append(a)
  srcid=a["source_annotation_id"];sid=a["image_id"]-1
  src=next(x for x in source_by[sid] if x["id"]==srcid)
  im=source_images[sid];original=decode_mask(src["segmentation"],im["height"],im["width"]);converted=decode_mask(a["segmentation"],im["height"],im["width"])
  require(np.array_equal(original,converted),"source pixel mismatch")
  box=mask_box(original)
  require(a["bbox"]==[box[0],box[1],box[2]-box[0],box[3]-box[1]] and a["area"]==int(original.sum()),"source geometry mismatch")
  require(a["area"]==stats[str(sid)]["annotation_areas"][str(srcid)],"metadata area disagreement")
  require(a["id"]==srcid+1 and a["iscrowd"]==src["iscrowd"] and a["attribute_ids"]==src.get("attribute_ids",[]),"source fields mismatch")
  checked+=1
  base={"image_id":a["image_id"],"source_annotation_id":srcid,"source_category_id":a["source_category_id"],"checker_type":"automated_source_record_check","date":"2026-11-18"}
  if 308 in src.get("attribute_ids",[]):
   evidence.append({**base,"basis":"source_attribute","target_label":"cutout_source_attribute","attribute_id":308,"attribute_name":"cutout","evidence_location":"source_subset.json#annotation="+str(srcid)+"/attribute_ids","status":"source_supported_not_visually_confirmed"})
  w,h=a["bbox"][2:];ratio=max(w/h,h/w);fill=a["area"]/(w*h)
  if a["source_category_id"] in SMALL and (ratio>=6 or fill<=.2):
   proxy={**base,"basis":"mask_geometry_proxy","target_label":"elongated_bbox_or_sparse_foreground_proxy","bbox_aspect_ratio":ratio,"mask_bbox_fill_ratio":fill,"rule":"bbox aspect >=6 OR foreground/bbox <=0.2","evidence_location":"annotations_coco.json#annotation="+str(a["id"]),"status":"screening_only_not_occlusion_or_boundary_quality"}
   evidence.append(proxy);geometry.append(proxy)
 eby=defaultdict(list)
 for e in evidence:eby[e["image_id"]].append(e)
 broad=set(stage["broad"]);rows=[]
 for im in manifest["images"]:
  key="train-"+im["source_image_id"];anns=by[im["image_id"]]
  targets=[]
  for a in anns:
   for label in sorted(hit_records([a])):
    targets.append({"source_annotation_id":a["source_annotation_id"],"source_category_id":a["source_category_id"],"target_label":label,"area_pixels":a["area"],"relative_area":a["area"]/(im["width"]*im["height"]),"basis":"source_attribute" if label=="complex_boundary_source" else "mask_geometry_proxy","evidence_location":"annotations_coco.json#annotation="+str(a["id"]),"checker_type":"automated_source_mask_statistics","date":"2026-11-18"})
  row={"image_id":im["image_id"],"source_dataset":"fashionpedia","source_split":"official_train2020","source_image_id":im["source_image_id"],"group_id":im["group_id"],"sha256":im["sha256"],"original_url":source_images[int(im["source_image_id"])].get("original_url"),"phase":stage["phase"],"sampling":"broad" if key in broad else "targeted","sampling_reason":sorted(hit_records(anns)) or ["broad_eight_class_coverage"],"source_annotation_ids":[a["source_annotation_id"] for a in anns],"target_evidence":targets,"boundary_evidence":eby[im["image_id"]] or [{"basis":"unknown","target_label":"complex_boundary","status":"not_assessed","image_id":im["image_id"],"source_annotation_id":None,"evidence_location":None,"checker_type":"none","date":"2026-11-18"}],"human_review_status":"not_human_reviewed"}
  rows.append(row)
 require({r["image_id"] for r in rows}==set(images),"sidecar coverage")
 write_json(delta/"sampling_sidecar.json",rows)
 write_json(delta/"boundary_evidence.json",evidence)
 report={"status":"passed","new_images":len(images),"source_masks_pixel_identical":checked,"geometry_and_source_fields_checked":checked,"old_masks_decoded":0,"source_annotation_sha256":sha256(SOURCE),"manifest_sha256":sha256(delta/"image_manifest.json"),"annotations_sha256":sha256(delta/"annotations_coco.json"),"source_supported_complex_images":len({e["image_id"] for e in evidence if e["basis"]=="source_attribute"}),"geometry_proxy_images":len({e["image_id"] for e in geometry}),"all_source_annotations_retained_for_selected_images":True,"human_confirmed_images":0}
 write_json(delta/"source_fidelity_report.json",report);print("SOURCE_CHECK",stage["phase"],checked,flush=True)
def copy_asset_tree(source,dest,known=None):
 sums={}
 for directory in ("images","masks"):
  (dest/directory).mkdir()
  for p in sorted((source/directory).iterdir()):
   rel=directory+"/"+p.name;shutil.copyfile(p,dest/rel);h=sha256(dest/rel)
   if known is not None:require(h==known[rel],"copied asset differs "+rel)
   sums[rel]=h
 return sums
def merge(parent,delta,out,phase):
 require(not out.exists(),"staging already exists; inspect before resume");out.mkdir()
 known=read_json(parent/"SHA256SUMS.json") if (parent/"SHA256SUMS.json").exists() else read_json(parent/"asset_hashes.json")
 sums=copy_asset_tree(parent,out,known)
 for directory in ("images","masks"):
  for p in sorted((delta/directory).iterdir()):
   rel=directory+"/"+p.name;require(not (out/rel).exists(),"asset collision");shutil.copyfile(p,out/rel);sums[rel]=sha256(out/rel)
   require(sums[rel]==sha256(p),"new copied asset mismatch")
 for file,fields in [("image_manifest.json",["images"]),("annotations_coco.json",["images","annotations"]),("candidates_gt.json",["images"]),("source_subset.json",["images","annotations"])]:
  obj=read_json(parent/file);new=read_json(delta/file)
  for field in fields:obj[field]+=new[field]
  if file=="image_manifest.json":obj["manifest_id"]=out.name
  if file=="candidates_gt.json":
   obj["candidate_set_id"]=out.name+"-gt";obj["image_manifest"]["sha256"]=sha256(out/"image_manifest.json")
   obj["provenance"]["producer_run_id"]=RUN.name;obj["provenance"]["annotation_manifest_sha256"]=sha256(out/"annotations_coco.json")
  if file=="annotations_coco.json":obj["info"]["description"]="Append-only staged official-train eight-class segmentation"
  write_json(out/file,obj)
 shutil.copyfile(PARENT/"mapping_config.json",out/"mapping_config.json")
 for name in ("queries_accepted.jsonl","queries_draft.jsonl"):require((parent/name).read_text()=="","parent unexpected language");(out/name).write_text("")
 verify_membership(parent,delta,out)
 def members(p):
  c=read_json(p/"annotations_coco.json")
  return {"image_ids":[i["id"] for i in c["images"]],"annotation_ids":[a["id"] for a in c["annotations"]]}
 old=members(PARENT);par=members(parent);new=members(delta);allmembers=members(out)
 oldsets={key:set(values) for key,values in old.items()}
 cumulative={key:[x for x in allmembers[key] if x not in oldsets[key]] for key in old}
 for name,obj in [("old10k_membership",old),("parent_membership",par),("added_subset_membership",new),("cumulative_added_membership",cumulative),("combined_membership",allmembers)]:write_json(out/(name+".json"),obj)
 side=read_json(delta/"sampling_sidecar.json");write_json(out/"sampling_sidecar.json",side)
 allside=(read_json(parent/"cumulative_sampling_sidecar.json") if phase==2 else [])+side
 write_json(out/"cumulative_sampling_sidecar.json",allside)
 require({x["image_id"] for x in side}==set(new["image_ids"]),"delta sidecar wrong")
 require({x["image_id"] for x in allside}==set(cumulative["image_ids"]),"cumulative sidecar wrong")
 require(len(allmembers["image_ids"])==10000+phase*5000,"combined count")
 require(len({x["group_id"] for x in read_json(out/"image_manifest.json")["images"]})==9976+phase*5000,"combined groups")
 write_json(out/"asset_hashes.json",sums)
 report={"status":"passed","original_parent":"fp_train10000_targeted_v1","direct_parent":parent.name,"parent_records_equal":True,"original_parent_records_equal":True,"parent_assets_destination_hash_checked":len(known.keys() & sums.keys()),"old_source_pixels_decoded":0,"images":len(allmembers["image_ids"]),"new_images":len(new["image_ids"]),"instances":len(allmembers["annotation_ids"]),"new_instances":len(new["annotation_ids"]),"sidecar_exactly_delta":True,"formal_freeze":False}
 # Directly verify original10k too, including phase2.
 for file,field,key in [("image_manifest.json","images","image_id"),("annotations_coco.json","annotations","id"),("source_subset.json","annotations","id")]:
  comb={r[key]:r for r in read_json(out/file)[field]}
  require(all(comb[r[key]]==r for r in read_json(PARENT/file)[field]),"original10k not preserved")
 write_json(out/"preservation_report.json",report)
 print("MERGED",out.name,report,flush=True)
def build():
 data,by=sources();config=read_json(PARENT/"mapping_config.json");mapping={r["source_id"]:r for r in config["fashionpedia"]}
 sources_dict={"train":{"data":data,"by":by,"hash":sha256(SOURCE)}};images={im["id"]:im for im in data["images"]}
 inv=read_json(RUN/"eligible_inventory.json");stages=read_json(RUN/"selected.json")["stages"]
 cats=read_json(REPO/"configs/categories.json")["categories"]
 parent=PARENT
 for stage in stages:
  n=stage["phase"];delta=RUN/f"fp_added5000_stage{n}_20261118";out=RUN/f"fp_train{10000+n*5000}_targeted_v1"
  if not (delta/"source_fidelity_report.json").exists():
   if not delta.exists():build_package(delta,"train",stage["keys"],inv,sources_dict,mapping,config,cats)
   require((delta/"validation_report.json").exists(),"incomplete delta build; inspect staging")
   sidecar_and_source_check(delta,stage,by,images)
  if not out.exists():merge(parent,delta,out,n)
  parent=out
 reports={"old10k":summarize(PARENT)}
 for n in (1,2):
  reports[f"delta{n}"]=summarize(RUN/f"fp_added5000_stage{n}_20261118")
  reports[f"combined{10000+n*5000}"]=summarize(RUN/f"fp_train{10000+n*5000}_targeted_v1")
  side=read_json(RUN/f"fp_added5000_stage{n}_20261118/sampling_sidecar.json")
  actual=Counter(label for row in side for label in set(row["sampling_reason"]))
  reports[f"delta{n}"]["requested_goals"]={g:{"goal_reference_phase1":v,"actual":actual[g],"gap":max(0,v-actual[g])} for g,v in GOALS.items()}
  reports[f"delta{n}"]["boundary_evidence"]=read_json(RUN/f"fp_added5000_stage{n}_20261118/source_fidelity_report.json")
 write_json(RUN/"distribution_report.json",reports)
 with (RUN/"distribution_summary.csv").open("w",newline="") as f:
  fields=["subset","taxonomy","category_id","name","instances","images","groups","small_instances_le1024","small_images","extreme_instances_le256","extreme_images"];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for subset,d in reports.items():
   for tax in ("project_categories","source_categories"):
    for cid,row in d[tax].items():w.writerow({"subset":subset,"taxonomy":tax,"category_id":cid,**{k:row[k] for k in fields[3:]}})
 print("BUILD_COMPLETE_PENDING_VISUAL",flush=True)
def preview():
 output=RUN/"visual_preview";output.mkdir(exist_ok=True);sample=[]
 for phase in (1,2):
  delta=RUN/f"fp_added5000_stage{phase}_20261118";side=read_json(delta/"sampling_sidecar.json");co=read_json(delta/"annotations_coco.json");anns={a["source_annotation_id"]:a for a in co["annotations"]};ims={i["id"]:i for i in co["images"]}
  rng=random.Random(SEED+phase)
  for goal in GOALS:
   available=[r for r in side if goal in r["sampling_reason"]];rng.shuffle(available)
   for row in available[:2]:
    evidence=next(t for t in row["target_evidence"] if t["target_label"]==goal);a=anns[evidence["source_annotation_id"]];im=ims[row["image_id"]]
    sample.append({"phase":phase,"stratum":goal,"image_id":row["image_id"],"source_annotation_id":a["source_annotation_id"],"project_annotation_id":a["id"],"file_name":im["file_name"],"area":a["area"],"source_category":a["source_category_name"],"delta":str(delta)})
  proxy_rows=[r for r in side if any(e["basis"]=="mask_geometry_proxy" for e in r["boundary_evidence"])]
  rng.shuffle(proxy_rows)
  for row in proxy_rows[:2]:
   e=next(e for e in row["boundary_evidence"] if e["basis"]=="mask_geometry_proxy");a=anns[e["source_annotation_id"]];im=ims[row["image_id"]]
   sample.append({"phase":phase,"stratum":"geometry_proxy","image_id":row["image_id"],"source_annotation_id":a["source_annotation_id"],"project_annotation_id":a["id"],"file_name":im["file_name"],"area":a["area"],"source_category":a["source_category_name"],"delta":str(delta)})
 # Each tile: full original with target box + enlarged target overlay; no test/old images.
 for page in range((len(sample)+3)//4):
  canvas=Image.new("RGB",(1600,1200),"white");draw=ImageDraw.Draw(canvas)
  for j,row in enumerate(sample[page*4:page*4+4]):
   delta=Path(row["delta"]);c=read_json(delta/"annotations_coco.json");a=next(a for a in c["annotations"] if a["id"]==row["project_annotation_id"]);im=next(i for i in c["images"] if i["id"]==row["image_id"])
   image=Image.open(delta/im["file_name"]).convert("RGB");mask=decode_mask(a["segmentation"],im["height"],im["width"]);box=mask_box(mask)
   origin=image.copy();ImageDraw.Draw(origin).rectangle([box[0],box[1],box[2]-1,box[3]-1],outline="red",width=3)
   arr=np.array(image);sel=mask.astype(bool);arr[sel]=(arr[sel]*.55+np.array([255,0,0])*.45).astype(np.uint8);overlay=Image.fromarray(arr)
   pad=max(20,int(max(box[2]-box[0],box[3]-box[1])*.15));crop=(max(0,box[0]-pad),max(0,box[1]-pad),min(im["width"],box[2]+pad),min(im["height"],box[3]+pad))
   cropim=overlay.crop(crop)
   x=(j%2)*800;y=(j//2)*600;draw.text((x+8,y+5),f"{page*4+j+1}: phase{row['phase']} {row['stratum']} im{row['image_id']} ann{a['source_annotation_id']}",fill="black")
   draw.text((x+8,y+22),f"{row['source_category']} / area={a['area']} / red overlay is original source mask",fill="black")
   for off,pic in [(0,origin),(400,cropim)]:
    pic=ImageOps.contain(pic,(390,550));canvas.paste(pic,(x+off+(400-pic.width)//2,y+45+(550-pic.height)//2))
   row["preview_page"]=f"page_{page+1:02d}.jpg";row["tile"]=j+1
  canvas.save(output/f"page_{page+1:02d}.jpg",quality=92)
 write_json(output/"sample_manifest.json",sample);print("PREVIEW_READY",len(sample),flush=True)
def attach_visual_audit():
 audit=read_json(RUN/"visual_audit.json");by=defaultdict(list)
 for record in audit["observations"]:by[(record["phase"],record["image_id"])].append(record)
 cumulative=[];attached=0
 stages=read_json(RUN/"selected.json")["stages"];stats=read_json(RUN/"metadata_stats.json")
 for phase in (1,2):
  delta=RUN/f"fp_added5000_stage{phase}_20261118";out=RUN/f"fp_train{10000+phase*5000}_targeted_v1"
  rows=read_json(delta/"sampling_sidecar.json")
  stage=stages[phase-1];modes={k:"broad_initial" for k in stage["broad"]}
  counts=Counter(g for k in stage["broad"] for g in stats[k[6:]]["hits"])
  for k in stage["targeted"]:
   hits=stats[k[6:]]["hits"];score=sum(max(0,GOALS[g]-counts[g])/GOALS[g] for g in hits)
   modes[k]="quota_deficit" if score>0 else "broad_backfill";counts.update(hits)
  for row in rows:
   row["selection_mode"]=modes["train-"+row["source_image_id"]]
   row["sampling_field_semantics"]="initial broad/targeted slot; selection_mode distinguishes quota choices from broad backfill"

   row["visual_checks"]=[{"basis":"visual_observation","checker_type":"AI_Codex","date":"2026-11-18","image_id":r["image_id"],"source_annotation_id":r["source_annotation_id"],"target_label":r["stratum"],"observation":r["observation"],"assessment":r["assessment"],"evidence_location":"audit/visual_preview/"+r["preview_page"]+"#tile="+str(r["tile"]),"not_human_confirmed":True} for r in by[(phase,row["image_id"])]]
  for row in rows:
   require(all(v["source_annotation_id"] in row["source_annotation_ids"] for v in row["visual_checks"]),"visual observation references another instance")
   attached+=len(row["visual_checks"])
  write_json(delta/"sampling_sidecar.json",rows);write_json(out/"sampling_sidecar.json",rows)
  cumulative+=rows;write_json(out/"cumulative_sampling_sidecar.json",cumulative)
 require(attached==len(audit["observations"]),"orphan visual observation")
def publish():
 require(read_json(RUN/"visual_audit.json")["status"]=="ai_spot_checked","visual audit not done")
 attach_visual_audit()
 protection=read_json(RUN/"protection_inputs.json")
 require({str(p) for p in SHARED.glob("fp_language_dev*/image_manifest.json")}<=set(protection),"new dev protection appeared; recheck selection")
 for f,h in protection.items():require(sha256(f)==h,"protected manifest changed")
 for package,files in read_json(RUN/"input_preflight.json")["protected_pins"].items():
  for name,h in files.items():require(sha256(SHARED/package/name)==h,"protected fixed pin changed")
 for f,h in PINS.items():require(sha256(PARENT/f)==h,"parent changed")
 commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip()
 auditnames=["input_preflight.json","feasibility_metadata.json","feasibility_checked.json","duplicate_audit.json","bridge_audit.json","protected_fingerprints.json","protection_inputs.json","distribution_report.json","distribution_summary.csv","download_provenance.json","visual_audit.json","independent_checks.json","focused_tests.txt"]
 releases=[]
 for n in (1,2):
  out=RUN/f"fp_train{10000+n*5000}_targeted_v1";delta=RUN/f"fp_added5000_stage{n}_20261118"
  require(read_json(RUN/"independent_checks.json")["status"]=="passed","independent checks missing")
  audit=out/"audit";audit.mkdir(exist_ok=True)
  for name in auditnames:shutil.copyfile(RUN/name,audit/name)
  shutil.copyfile(REPO/"docs/handoffs/data_eval/language_targeted_20261117_evidence/dev_exposure_registry.json",audit/"dev_exposure_registry.json")
  for phase in range(1,n+1):
   dp=RUN/f"fp_added5000_stage{phase}_20261118"
   for name in ["source_fidelity_report.json","validation_report.json","boundary_evidence.json","conversion_report.json"]:shutil.copyfile(dp/name,audit/f"stage{phase}_{name}")
  shutil.copytree(RUN/"visual_preview",audit/"visual_preview",dirs_exist_ok=True)
  # Parent's hash-bound existing pixel/mask audit: reference original evidence, no re-decoding.
  shutil.copyfile(PARENT/"preservation_report.json",audit/"original10k_preservation_report.json")
  direct_parent=PARENT if n==1 else SHARED/"fp_train15000_targeted_v1"
  report=read_json(out/"preservation_report.json");checks=read_json(delta/"source_fidelity_report.json")
  release={"schema_version":3,"illustrative_only":False,"release_id":out.name,"owner":"data_eval","executed_by":"wkq","code_commit":commit,"parent":direct_parent.name,"original_parent":PARENT.name,"parent_manifest_sha256":sha256(direct_parent/"image_manifest.json"),"original_parent_pins":PINS,"source_annotation_sha256":sha256(SOURCE),"mapping_sha256":sha256(out/"mapping_config.json"),"seed":SEED,"phase":n,"images":report["images"],"instances":report["instances"],"groups":9976+n*5000,"parent_images":10000+(n-1)*5000,"original_parent_images":10000,"new_groups":5000,"new_images":5000,"new_instances":checks["source_masks_pixel_identical"],"cumulative_new_images":n*5000,"sampling":{"broad":2500,"targeted":2500},"sampling_semantics":"slot budget; priority slots backfill in fixed order once no available deficit can be filled","actual_selection_modes":dict(Counter(r["selection_mode"] for r in read_json(delta/"sampling_sidecar.json"))),"technical_status":"passed_for_new_increment","legacy_bridge_clues":read_json(RUN/"bridge_audit.json")["active_frozen_train_dev_test_exposure_bridge_count"],"legacy_bridge_status":"unconfirmed clues; all involved new candidates quarantined; old splits unchanged","coverage_status":"partial_see_distribution_report","human_review_status":"not_exhaustive; AI stratified visual spot check only","formal_freeze":False,"training_performed":False,"model_smoke_performed":False,"new_language_queries":0,"consumer_acceptance":"pending_lcy_incremental_CPU_loader_check","delta_cpu_view":str(delta),"source_fidelity_reports":[f"audit/stage{p}_source_fidelity_report.json" for p in range(1,n+1)],"sidecar_scope":"sampling_sidecar equals direct phase delta; cumulative_sampling_sidecar equals all additions over original10k","request":"https://github.com/rickyliuchunyu/comp4471-fashion-grounding/blob/64e504ed29beb843289931a28349d9da3228d630/docs/handoffs/segmentation/train15000_20000_expansion_request_20261118.md","limitations":["Source cutout attribute is supporting evidence, not human-verified mask quality.","pHash grouping does not prove exhaustive same-product detection.","Small-bag availability insufficient; second-stage quotas are reported separately, not assumed doubled.","Original parent duplicate structure preserved; only new groups are independent."]}
  write_json(out/"RELEASE.json",release)
  # Already checked copied assets once. Hash metadata/evidence only now.
  sums=read_json(out/"asset_hashes.json")
  for p in sorted(out.rglob("*")):
   if p.is_file() and p.relative_to(out).parts[0] not in ("images","masks") and p.name!="SHA256SUMS.json":sums[p.relative_to(out).as_posix()]=sha256(p)
  write_json(out/"SHA256SUMS.json",sums)
  final=SHARED/out.name;require(not final.exists(),"release exists")
  out.rename(final) # atomic same data disk; no third copy/hash pass.
  releases.append({"package":str(final),"manifest_sha256":sha256(final/"image_manifest.json"),"annotations_sha256":sha256(final/"annotations_coco.json"),"checksums_sha256":sha256(final/"SHA256SUMS.json"),"release":release})
 write_json(RUN/"release_index.json",releases)
 write_json(RUN/"run_metadata.json",{"run_id":RUN.name,"status":"completed","owner":"data_eval","executed_by":"wkq","git_commit":commit,"python":sys.version,"executable":sys.executable,"platform":platform.platform(),"commands":["expand_segmentation_staged.py inventory/acquire/select","build_staged_segmentation.py build/preview/publish","verify_staged_segmentation.py"],"training_performed":False,"gpu_used":False,"evaluation":None,"artifacts":releases})
 print("PUBLISHED",[(r["package"],r["release"]["images"]) for r in releases],flush=True)
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("step",choices=["build","preview","publish"]);a=p.parse_args();globals()[a.step]()
