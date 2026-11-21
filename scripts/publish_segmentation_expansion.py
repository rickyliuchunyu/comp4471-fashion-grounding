"""Publish verified append-only segmentation data; never replace an existing version."""
import argparse,shutil,subprocess,sys,platform
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from expand_segmentation import SHARED,FROZEN,frozen_snapshot
from fashion_grounding.data.schema import read_json,write_json,sha256,require
def publish(run):
 out=run/"fp_train10000_targeted_v1"; final=SHARED/out.name
 require(not final.exists(),"release already exists")
 require(read_json(run/"preservation_report.json")["status"]=="passed","preservation not verified")
 require(read_json(run/"visual_audit.json")["status"]=="spot_checked","visual review incomplete")
 require("PIPELINE_COMPLETE_PENDING_VISUAL_REVIEW_AND_PUBLICATION" in (run/"pipeline.log").read_text(),"pipeline incomplete")
 require(frozen_snapshot()==read_json(run/"frozen_before.json"),"frozen inputs changed")
 audit=out/"audit";audit.mkdir(exist_ok=True)
 names=["distribution_report.json","distribution_summary.csv","eligible_pool_report.json","duplicate_audit.json","independent_duplicate_check.json","pool_mask_check.json","download_provenance.json","visual_audit.json","preservation_report.json"]
 for name in names:shutil.copy2(run/name,audit/name)
 for name,src in [("parent_consumer_check_report.json",SHARED/FROZEN[0]),("added_consumer_check_report.json",run/"fp_added5000_targeted_v1")]:
  shutil.copy2(src/"consumer_check_report.json",audit/name)
 for name in ("class_preview","small_preview"):shutil.copytree(run/name,audit/name,dirs_exist_ok=True)
 old=read_json(SHARED/FROZEN[0]/"annotations_coco.json")
 added=read_json(run/"fp_added5000_targeted_v1"/"annotations_coco.json")
 for name,data in (("old_subset_membership.json",old),("added_subset_membership.json",added)):
  write_json(out/name,{"image_ids":[x["id"] for x in data["images"]],"annotation_ids":[x["id"] for x in data["annotations"]],"id_mapping":"Official train image and annotation IDs + 1; parent project IDs unchanged; every annotation retains source_annotation_id","parent":"fp_train5000_v1" if name.startswith("old") else None})
 commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
 release={"release_id":out.name,"owner":"data_eval","executed_by":"wkq","code_commit":commit,"parent":"fp_train5000_v1","source_annotation_sha256":"f6af94fb8712bd8170a49c857e842e1227e6806c4f33f32340db0c849133e006","seed":2026,"old_images":5000,"new_images":5000,"sampling":{"broad":3500,"targeted":1500},"request":"https://github.com/rickyliuchunyu/comp4471-fashion-grounding/blob/149ddf7/docs/handoffs/segmentation/train10000_expansion_request_20261115.md","frozen_inputs":read_json(run/"frozen_before.json"),"training_performed":False,"consumer_model_validation":"pending lcy loader acceptance","formal_freeze":False,"language_queries_added":0,"limitations":["Original parent duplicate groups are preserved; only additions are newly isolated.","Targeted class/size sampling is not a pure dataset-size ablation.","Spot review is not exhaustive human boundary certification; original source masks are preserved.","First unpublished selection was superseded after a stronger four-orientation audit; original staging outputs retained."],"source_pixel_check":"Unchanged parent evidence plus independent check of all added masks; see audit"}
 write_json(out/"RELEASE.json",release)
 sums={p.relative_to(out).as_posix():sha256(p) for p in sorted(out.rglob("*")) if p.is_file() and p.name!="SHA256SUMS.json"}
 write_json(out/"SHA256SUMS.json",sums)
 stage=SHARED/("."+out.name+"-staging-20261116")
 require(not stage.exists(),"staging exists; inspect instead of overwrite")
 shutil.copytree(out,stage)
 for name,value in sums.items():require(sha256(stage/name)==value,"publication copy mismatch: "+name)
 require(sha256(stage/"SHA256SUMS.json")==sha256(out/"SHA256SUMS.json"),"checksum index mismatch")
 stage.rename(final)
 metadata={"record_schema_version":1,"run_id":run.name,"experiment_id":"data_preparation","owner":"data_eval","executed_by":"wkq","status":"completed","git_commit":commit,"split":"train","candidate_source":"gt","seed":2026,"environment":{"python":sys.version,"platform":platform.platform(),"executable":sys.executable},"command":["driver.py","report_expansion_pool.py --run "+str(run),"verify_segmentation_expansion.py --run "+str(run),"publish_segmentation_expansion.py --run "+str(run)],"artifacts":{"package":str(final),"image_manifest":{"path":str(final/"image_manifest.json"),"sha256":sha256(final/"image_manifest.json")},"annotations":{"path":str(final/"annotations_coco.json"),"sha256":sha256(final/"annotations_coco.json")},"checksums":{"path":str(final/"SHA256SUMS.json"),"sha256":sha256(final/"SHA256SUMS.json")}},"evaluation":None,"evaluation_not_applicable":"CPU data preparation only; no model training or performance metrics","parent_run_ids":["20261112-large-data-regrouped"],"superseded_unpublished_staging":str(run.parent/"20261116-train10000")}
 write_json(run/"run_metadata.json",metadata)
 print(str(final));print(metadata["artifacts"])
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);a=p.parse_args();publish(a.run)
