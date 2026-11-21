"""Build an immutable draft package from visually authored bilingual query specifications."""
import argparse,copy,json,shutil
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from fashion_grounding.data.schema import read_json,write_json,sha256,require,validate_package

def build(spec_path, output):
 spec=read_json(spec_path);require(not output.exists(),"choose a new output version")
 coco_images={};ims={};anns={};candidate_images={};source_images={};source_anns={};roots={};licenses={};source_categories=None;categories=None;mapping=None
 for location in spec["source_packages"]:
  root=Path(location);validate_package(root)
  manifest=read_json(root/"image_manifest.json");coco=read_json(root/"annotations_coco.json");cs=read_json(root/"candidates_gt.json");src=read_json(root/"source_subset.json")
  for row in manifest["images"]:
   require(row["image_id"] not in ims,"duplicate input image")
   ims[row["image_id"]]=row;roots[row["image_id"]]=root
  require(not (set(anns) & {a["id"] for a in coco["annotations"]}),"duplicate input annotation")
  anns.update({a["id"]:a for a in coco["annotations"]})
  coco_images.update({im["id"]:im for im in coco["images"]})
  candidate_images.update({r["image_id"]:r for r in cs["images"]})
  source_images.update({r["id"]:r for r in src["images"]});source_anns.update({r["id"]:r for r in src["annotations"]})
  licenses.update({r["id"]:r for r in coco["licenses"]});source_categories=src["categories"];categories=coco["categories"]
  current=read_json(root/"mapping_config.json")
  require(mapping is None or mapping==current,"mapping mismatch");mapping=current
 used=sorted({q["image_id"] for q in spec["drafts"]}|{d["image_id"] for d in spec.get("disputes",[])})
 output.mkdir(parents=True);(output/"images").mkdir();(output/"masks").mkdir()
 chosen_ims=[];chosen_anns=[];chosen_cs=[]
 for iid in used:
  require(iid in ims and ims[iid]["split"]=="train","draft batch requires known training images")
  im=copy.deepcopy(ims[iid]);root=roots[iid]
  shutil.copyfile(root/im["file_name"],output/im["file_name"]);chosen_ims.append(im)
  chosen_anns.extend(a for a in anns.values() if a["image_id"]==iid)
  cs=copy.deepcopy(candidate_images[iid])
  for c in cs["candidates"]:shutil.copyfile(root/c["mask_reference"],output/c["mask_reference"])
  chosen_cs.append(cs)
 common={"schema_version":3,"illustrative_only":False}
 write_json(output/"image_manifest.json",{**common,"manifest_id":output.name,"mapping_version":mapping["mapping_version"],"images":chosen_ims})
 write_json(output/"annotations_coco.json",{**common,"info":{"description":"Visually authored training drafts; pending human review"},"licenses":list(licenses.values()),"images":[coco_images[im["image_id"]] for im in chosen_ims],"annotations":sorted(chosen_anns,key=lambda a:a["id"]),"categories":categories})
 write_json(output/"candidates_gt.json",{**common,"candidate_set_id":output.name+"-gt","candidate_source":"gt","image_manifest":{"path":"image_manifest.json","sha256":sha256(output/"image_manifest.json")},"provenance":{"producer_run_id":"wkq-grounded-query-drafts-v1","checkpoint_sha256":None,"decoder_config_sha256":None,"annotation_manifest_sha256":sha256(output/"annotations_coco.json")},"images":chosen_cs})
 raw_ids={int(im["source_image_id"]) for im in chosen_ims}
 write_json(output/"source_subset.json",{"info":{"description":"Official training subset; source IDs unchanged"},"licenses":list(licenses.values()),"categories":source_categories,"images":[source_images[i] for i in sorted(raw_ids)],"annotations":[a for a in source_anns.values() if a["image_id"] in raw_ids]})
 write_json(output/"mapping_config.json",mapping)
 rows=[];seen=set()
 for item in spec["drafts"]:
  sid=item["semantic_query_id"];require(sid not in seen,"duplicate semantic ID");seen.add(sid)
  require(item["target_instance_id"] in anns,"unknown target")
  require(bool(item["query_types"]) and set(item["query_types"]) <= {"category","spatial","appearance","combined"},"invalid query types")
  target=anns[item["target_instance_id"]];iid=item["image_id"]
  require(target["image_id"]==iid and not target["iscrowd"],"invalid target")
  for rival in item.get("competing_instance_ids",[]):
   require(rival in anns and anns[rival]["image_id"]==iid and rival!=target["id"] and not anns[rival]["iscrowd"] and anns[rival]["category_id"]==target["category_id"],"invalid rival")
  for lang in ("zh","en"):
   require(bool(item[lang].strip()),"empty text")
   rows.append({**common,"query_id":sid+"-"+lang,"semantic_query_id":sid,"contrast_set_id":item.get("contrast_set_id"),"image_id":iid,"source_dataset":"fashionpedia","split":"train","query":item[lang],"language":lang,"target_instance_id":target["id"],"target_category_id":target["category_id"],"same_category_instance_count":sum(a["image_id"]==iid and a["category_id"]==target["category_id"] and not a["iscrowd"] for a in chosen_anns),"query_types":item["query_types"],"spatial_reference":"image_view","annotation_version":output.name,"annotator":"codex_visual_annotation","reviewer":None,"review_status":"needs_revision","target_source_category_id":target["source_category_id"],"target_source_category_name":target["source_category_name"],"same_source_category_instance_count":sum(a["image_id"]==iid and a["source_category_id"]==target["source_category_id"] and not a["iscrowd"] for a in chosen_anns),"authorship":"machine_authored_after_visual_inspection_pending_human_review","template_family":None,"paraphrase_group_id":item.get("paraphrase_group_id")})
 (output/"queries_draft.jsonl").write_text("".join(json.dumps(q,ensure_ascii=False)+"\n" for q in rows),encoding="utf-8")
 (output/"queries_accepted.jsonl").write_text("")
 write_json(output/"quarantine_report.json",{"images":spec.get("quarantined_images",[]),"action":"Excluded from this language package; source data unchanged; no automatic mask merge."})
 write_json(output/"annotation_notes.json",{"authorship":"codex_visual_annotation","human_review":"pending","drafts":spec["drafts"],"disputes":spec.get("disputes",[])})
 types=Counter("+".join(sorted(q["query_types"])) for q in spec["drafts"])
 summary={**common,"images":len(chosen_ims),"instances":len(chosen_anns),"semantic_queries":len(spec["drafts"]),"language_records":len(rows),"accepted":0,"query_types":dict(types),"same_category_competition_semantic_queries":sum(q["same_category_instance_count"]>=2 for q in rows if q["language"]=="zh"),"target_categories":dict(Counter(q["target_category_id"] for q in rows if q["language"]=="zh")),"dispute_proposals":len(spec.get("disputes",[])),"scope":"Training-only authored drafts, no benchmark result, no final-test claims","source_manifest_sha256":[sha256(Path(p)/"image_manifest.json") for p in spec["source_packages"]],"annotation_spec_sha256":sha256(spec_path)}
 write_json(output/"annotation_summary.json",summary)
 write_json(output/"validation_report.json",validate_package(output))
 return summary

if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--spec",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
 print(json.dumps(build(a.spec,a.output),ensure_ascii=False,indent=2))
