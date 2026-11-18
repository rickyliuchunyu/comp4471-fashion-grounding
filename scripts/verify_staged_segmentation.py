"""Metadata-only independent checks; never decode or rescan original10k/test assets."""
import json,sys
from collections import Counter
from pathlib import Path
from expand_segmentation_staged import *
from build_staged_segmentation import verify_membership
def bridge_report(audit,protected):
 protected={x["key"]:x for x in protected}
 roles={}
 for name,label in [("fp_dev250_v1","fixed_dev250"),("fp_test400_holdout_v1","fixed_test400")]:
  for im in read_json(SHARED/name/"image_manifest.json")["images"]:roles["val-"+im["source_image_id"]]=label
 for row in read_json(REPO/"docs/handoffs/data_eval/language_targeted_20261117_evidence/dev_exposure_registry.json")["images"]:
  roles["val-"+str(row["image_id"]-10**9-1)]=row["status"]
 oldkeys={"train-"+im["source_image_id"] for im in read_json(PARENT/"image_manifest.json")["images"]}
 components={tuple(sorted(r["component"])) for r in audit["excluded"] if "component" in r}
 bridges=[]
 for comp in sorted(components):
  train=[k for k in comp if k in protected and k.startswith("train-")];val=[k for k in comp if k in protected and k.startswith("val-")]
  if train and val:
   bridges.append({"component":list(comp),"frozen_train_keys":train,"protected_val_keys":val,"actual_old10k_keys":[k for k in train if k in oldkeys],"validation_roles":{k:roles.get(k,"associated_group_member_not_fixed_dev_test_or_exposed50") for k in val},"active_split_bridge":bool(set(train)&oldkeys and any(k in roles for k in val)),"action":"all_new_members_quarantined; frozen_groups_unchanged","interpretation":"perceptual/URL component clue, not visually confirmed leakage"})
 report={"count":len(bridges),"active_frozen_train_dev_test_exposure_bridge_count":sum(r["active_split_bridge"] for r in bridges),"cross_train_validation_components":bridges,"test_pixels_viewed":False,"review_status":"active bridge clues require separate data-owner disposition; no old split changed"}
 write_json(RUN/"bridge_audit.json",report)
 return report
def main():
 selection=read_json(RUN/"selected.json");inv=read_json(RUN/"eligible_inventory.json");protected=read_json(RUN/"protected_fingerprints.json")
 audit=read_json(RUN/"duplicate_audit.json");blocked={r["key"] for r in audit["excluded"]};oldkeys={x["key"] for x in protected}
 keys=[k for st in selection["stages"] for k in st["keys"]];selected=set(keys)
 require(len(keys)==10000==len(selected),"selected duplicate/count")
 require(not selected&oldkeys and not selected&blocked,"selected protected/blocked")
 require(not any(e["a"] in selected and (e["b"] in selected or e["b"] in oldkeys) or e["b"] in selected and e["a"] in oldkeys for e in audit["edges"]),"retained duplicate edge")
 # Independent vector check only selected-new vs protected/new, not old-old.
 old=list(protected);new=[inv[k] for k in keys];allrows=old+new
 h=np.array([x["phash"] for x in allrows],dtype=np.uint64);f=np.array([x["phash_flip"] for x in allrows],dtype=np.uint64)
 hashes={x["sha256"] for x in old};urls={specific_source_url(x.get("original_url")) for x in old}-{None}
 for i in range(len(old),len(allrows)):
  x=allrows[i];u=specific_source_url(x.get("original_url"))
  require(x["sha256"] not in hashes and (not u or u not in urls),"new SHA/URL conflict")
  hashes.add(x["sha256"])
  if u:urls.add(u)
  d=np.minimum.reduce([np.bitwise_count(h[:i]^h[i]),np.bitwise_count(f[:i]^h[i]),np.bitwise_count(h[:i]^f[i]),np.bitwise_count(f[:i]^f[i])])
  require(not np.any(d<=6),"new pHash conflict")
 reports=[]
 for st in selection["stages"]:
  n=st["phase"];delta=RUN/f"fp_added5000_stage{n}_20261118";combined=RUN/f"fp_train{10000+n*5000}_targeted_v1";parent=PARENT if n==1 else RUN/"fp_train15000_targeted_v1"
  verify_membership(parent,delta,combined)
  co=read_json(delta/"annotations_coco.json");raw=read_json(delta/"source_subset.json");side=read_json(delta/"sampling_sidecar.json");mapping={r["source_id"]:r for r in read_json(delta/"mapping_config.json")["fashionpedia"]}
  require({a["source_annotation_id"] for a in co["annotations"]}=={a["id"] for a in raw["annotations"] if mapping[a["category_id"]]["action"]=="map"},"source instance omitted")
  require(not any(a["category_id"]==11 for a in raw["annotations"]),"jumpsuit included")
  require(all(a["category_id"]==mapping[a["source_category_id"]]["project_category_id"] for a in co["annotations"]),"wrong mapping")
  require(len(side)==len(co["images"])==5000 and {r["image_id"] for r in side}=={i["id"] for i in co["images"]},"sidecar not exact")
  require(Counter(r["sampling"] for r in side)=={"broad":2500,"targeted":2500},"sampling allocation")
  evidence=read_json(delta/"source_fidelity_report.json")
  require(evidence["source_masks_pixel_identical"]==len(co["annotations"]) and evidence["annotations_sha256"]==sha256(delta/"annotations_coco.json") and evidence["manifest_sha256"]==sha256(delta/"image_manifest.json"),"unbound source evidence")
  require(read_json(delta/"validation_report.json")["status"]=="passed","delta schema invalid")
  reports.append({"phase":n,"new_images":5000,"new_annotations":len(co["annotations"]),"source_masks_checked":evidence["source_masks_pixel_identical"],"source_supported_boundary_images":evidence["source_supported_complex_images"],"proxy_images":evidence["geometry_proxy_images"]})
 for p,h in read_json(RUN/"protection_inputs.json").items():require(sha256(p)==h,"protected file changed")
 for p,h in PINS.items():require(sha256(PARENT/p)==h,"old10k changed")
 for package,files in read_json(RUN/"input_preflight.json")["protected_pins"].items():
  for name,h in files.items():require(sha256(SHARED/package/name)==h,"protected fixed pin changed")
 report={"status":"passed","new_images":len(keys),"protected_fingerprints":len(protected),"new_duplicate_conflicts":0,"new_vs_old_and_new_only":True,"old_old_comparisons":0,"phash_orientations":4,"phash_threshold":6,"source_identity_namespace":"fashionpedia/official_train2020/source_id vs fashionpedia/official_val2020/source_id","phases":reports,"parent_pins_and_protection_manifests_unchanged":True,"not_claimed":"No new exhaustive human review or proof that perceptual hashing finds all related products."}
 report["bridge_audit"]=bridge_report(audit,protected)
 write_json(RUN/"independent_checks.json",report);print(json.dumps(report,indent=2))
if __name__=="__main__":main()
