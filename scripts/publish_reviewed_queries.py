"""Explicit training-only publication; preserve rejected and exposed review evidence."""
import argparse,copy,json,shutil,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from fashion_grounding.data.schema import read_json,write_json,sha256,require,validate_package
from fashion_grounding.data.review import audit_review

def publish(source, review_path, output):
    validate_package(source)
    review=read_json(review_path)
    require(review.get("review_format_version")==2,"quality-check review v2 required")
    require(review["query_sha256"]==sha256(source/"queries_draft.jsonl"),"query hash mismatch")
    manifest=read_json(source/"image_manifest.json")
    require(all(i["split"]=="train" for i in manifest["images"]),"only training publication supported")
    queries=[json.loads(s) for s in (source/"queries_draft.jsonl").read_text().splitlines() if s.strip()]
    audit=audit_review(manifest,read_json(source/"annotations_coco.json")["annotations"],queries,review)
    decisions={r["semantic_query_id"]:r for r in review["rows"]}
    eligible=set()
    for row in audit["rows"]:
        d=decisions.get(row["semantic_query_id"],{})
        if (d.get("decision")=="human_selected" and row["target_agrees"] and
            all(d.get("checks",{}).get(k) is True for k in ("unique","translation","mask")) and
            d.get("edited_queries")==row["queries"] and not d.get("note","").strip()):
            eligible.add(row["semantic_query_id"])
    require(not output.exists(),"output exists")
    output.mkdir(parents=True)
    # Copy data only, not stale draft README, review UI, hashes, or reports.
    for name in ("images","masks"):
        shutil.copytree(source/name,output/name)
    for name in ("annotations_coco.json","mapping_config.json","source_subset.json","annotation_notes.json","quarantine_report.json"):
        shutil.copyfile(source/name,output/name)
    manifest["manifest_id"]=output.name;write_json(output/"image_manifest.json",manifest)
    candidates=read_json(source/"candidates_gt.json")
    candidates["candidate_set_id"]=output.name+"-gt"
    candidates["image_manifest"]["sha256"]=sha256(output/"image_manifest.json")
    write_json(output/"candidates_gt.json",candidates)
    accepted=[];pending=[]
    for original in queries:
        q=copy.deepcopy(original);q["annotation_version"]=output.name
        if q["semantic_query_id"] in eligible:
            require(q["annotator"]!=review["reviewer"],"self review")
            q.update(review_status="accepted",reviewer=review["reviewer"],
                review_evidence_sha256=sha256(review_path),
                review_mode="previously_exposed_training_rereview" if q["image_id"] in review.get("disclosed_image_ids",[]) else "human_target_and_quality_review")
            accepted.append(q)
        else:pending.append(q)
    for name,rows in (("queries_accepted.jsonl",accepted),("queries_draft.jsonl",pending)):
        (output/name).write_text("".join(json.dumps(q,ensure_ascii=False)+"\n" for q in rows),encoding="utf-8")
    summary={"scope":"training pilot; not independent test","accepted_semantics":len(eligible),"accepted_records":len(accepted),"pending_records":len(pending),"review_sha256":sha256(review_path),"parent_query_sha256":review["query_sha256"],"previously_exposed_accepted_records":sum(q["review_mode"].startswith("previously") for q in accepted),"old_disputes":"New query review does not retroactively change old decisions."}
    write_json(output/"pending_review.json",{"rows":[r for r in audit["rows"] if r["semantic_query_id"] not in eligible]})
    write_json(output/"review_release.json",summary)
    write_json(output/"validation_report.json",validate_package(output))
    return summary
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--source",type=Path,required=True);p.add_argument("--review",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    print(json.dumps(publish(a.source,a.review,a.output),indent=2))
