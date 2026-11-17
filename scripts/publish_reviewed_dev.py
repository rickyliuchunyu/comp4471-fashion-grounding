"""Freeze agreed dev review pairs before model inference; preserve all pending evidence."""
import argparse,copy,json,shutil
from pathlib import Path
from fashion_grounding.data.review import audit_review
from fashion_grounding.data.schema import read_json,sha256,require,write_json,validate_package

def publish(source,reviews,output):
    require(not output.exists(),"output exists")
    manifest=read_json(source/"image_manifest.json")
    require(all(im["split"]=="dev" for im in manifest["images"]),"dev-only publication")
    queries=[json.loads(l) for l in (source/"queries_draft.jsonl").read_text().splitlines() if l.strip()]
    merged=dict(review_format_version=2,manifest_id=manifest["manifest_id"],query_sha256=sha256(source/"queries_draft.jsonl"),reviewer=None,rows=[],disclosed_image_ids=[],source_files=[])
    seen=set()
    for f in sorted(reviews.glob("*human-review.json")):
        d=read_json(f)
        require(d["review_format_version"]==2 and d["manifest_id"]==merged["manifest_id"] and d["query_sha256"]==merged["query_sha256"],"review identity mismatch")
        if merged["reviewer"] is None:merged["reviewer"]=d["reviewer"]
        require(d["reviewer"]==merged["reviewer"],"mixed reviewers")
        for row in d["rows"]:
            require(row["semantic_query_id"] not in seen,"duplicate semantic review")
            seen.add(row["semantic_query_id"]);merged["rows"].append(row)
        merged["disclosed_image_ids"]+=d.get("disclosed_image_ids",[])
        merged["source_files"].append(dict(file=f.name,sha256=sha256(f)))
    require(seen=={q["semantic_query_id"] for q in queries},"review page missing or unknown")
    notes=read_json(source/"annotation_notes.json")
    disclosed={d["image_id"] for d in notes.get("disputes",[]) if d.get("prior_answer_exposure")}
    merged["disclosed_image_ids"]=sorted(set(merged["disclosed_image_ids"])|disclosed)
    audit=audit_review(manifest,read_json(source/"annotations_coco.json")["annotations"],queries,merged)
    eligible={row["semantic_query_id"] for row in audit["rows"] if row["status"]=="target_agreement_only" and all(row["checks"].values())}
    require(eligible,"no fully agreed pairs")
    output.mkdir(parents=True)
    for name in ("images","masks"):shutil.copytree(source/name,output/name)
    for name in ("annotations_coco.json","mapping_config.json","source_subset.json","annotation_notes.json"):shutil.copy2(source/name,output/name)
    manifest["manifest_id"]=output.name;write_json(output/"image_manifest.json",manifest)
    c=read_json(source/"candidates_gt.json");c["candidate_set_id"]=output.name+"-gt";c["image_manifest"]["sha256"]=sha256(output/"image_manifest.json");write_json(output/"candidates_gt.json",c)
    write_json(output/"human_review.json",merged);write_json(output/"review_audit.json",audit)
    accepted=[];pending=[]
    for row in queries:
        q=copy.deepcopy(row);q["annotation_version"]=output.name
        if q["semantic_query_id"] in eligible:
            require(q["annotator"]!=merged["reviewer"],"self review")
            q.update(review_status="accepted",reviewer=merged["reviewer"],review_mode="human_target_and_quality_review",review_evidence_sha256=sha256(output/"human_review.json"))
            accepted.append(q)
        else:pending.append(q)
    for name,rows in (("queries_accepted.jsonl",accepted),("queries_draft.jsonl",pending),("queries_parent.jsonl",queries)):
        (output/name).write_text("".join(json.dumps(q,ensure_ascii=False)+"\n" for q in rows),encoding="utf-8")
    summary=dict(scope="reviewed dev subset; pending labels excluded before predictions, not based on model success",
        parent=source.name,parent_query_sha256=merged["query_sha256"],review_files=len(merged["source_files"]),accepted_semantics=len(eligible),
        accepted_records=len(accepted),accepted_images=len({q["image_id"] for q in accepted}),pending_semantics=len(pending)//2,
        image_count=len(manifest["images"]),status_counts=audit["status_counts"],targets_or_text_changed=False,
        package_complete_human_acceptance=False)
    write_json(output/"review_release.json",summary);write_json(output/"validation_report.json",validate_package(output))
    require(c["images"]==read_json(source/"candidates_gt.json")["images"],"candidate content changed")
    write_json(output/"SHA256SUMS.json",{str(p.relative_to(output)):sha256(p) for p in sorted(output.rglob("*")) if p.is_file() and p.name!="SHA256SUMS.json"})
    return summary
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("source","reviews","output"):p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args();print(json.dumps(publish(a.source,a.reviews,a.output),ensure_ascii=False,indent=2))
