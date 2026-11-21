"""Build a train-only visual draft without changing partitions or source masks."""
import argparse, copy, json, re, shutil
from collections import Counter
from pathlib import Path
from build_dev_visual_review import dump, jsonl
from revise_dev_queries import pairs, coverage, features, DISALLOWED
from fashion_grounding.data.schema import read_json, require, sha256, validate_package

def prepare(manifest, coco, old, spec, index, version):
    require(all(i["split"] == "train" for i in manifest["images"]), "train-only package required")
    ps = pairs(old)
    require(all(q["split"] == "train" for q in old), "train-only queries required")
    require(set(spec["viewed_sheets"]) == {i["sheet"] for i in index}, "incomplete visual sheets")
    require({i["image_id"] for i in index} == {i["image_id"] for i in manifest["images"]}, "incomplete visual image coverage")
    shown = {(i["image_id"], t["id"]) for i in index for t in i["targets"]}
    require(all((q["image_id"], q["target_instance_id"]) in shown for q in old), "query target not visually covered")
    anns = {a["id"]: a for a in coco["annotations"]}
    overrides = spec["overrides"]
    require(set(map(int, overrides)) <= set(anns), "unknown override target")
    require(set(map(int, overrides)) <= {q["target_instance_id"] for q in old}, "unused override")
    rows, quarantined, audit = [], [], []
    sheet = {i["image_id"]: i["sheet"] for i in index}
    by_image = {}
    for a in anns.values():
        if not a.get("iscrowd",0): by_image.setdefault(a["image_id"], []).append(a)
    def emit(base, aid, texts, sid, action, reason=None):
        a = anns[aid]
        require(a["image_id"] == base["image_id"], "target/image mismatch")
        require(not a.get("iscrowd",0), "crowd target")
        require(set(texts) == {"zh","en"} and all(isinstance(t,str) and t.strip() for t in texts.values()), "bilingual text required")
        require(not any(re.search(DISALLOWED,t,re.I) for t in texts.values()), "fine/body-relation description")
        spatial = bool(re.search(r"左|右|中央|中间|下方|上方|\b(?:left|right|center|middle|lower|upper)\b", " ".join(texts.values()), re.I))
        fs = features({l:{"query":t} for l,t in texts.items()})
        appearance = bool(fs - {"left","right","middle","lower"})
        candidates = by_image[a["image_id"]]
        for lang in ("zh","en"):
            q = copy.deepcopy(base)
            q.update(query_id=sid+"-"+lang,semantic_query_id=sid,language=lang,query=texts[lang],
                     target_instance_id=aid,target_category_id=a["category_id"],
                     target_source_category_id=a["source_category_id"],target_source_category_name=a["source_category_name"],
                     same_category_instance_count=sum(x["category_id"]==a["category_id"] for x in candidates),
                     same_source_category_instance_count=sum(x["source_category_id"]==a["source_category_id"] for x in candidates),
                     annotation_version=version,annotator="codex_visual_audit_20261117",reviewer=None,review_status="needs_revision",
                     authorship="machine_visual_inspection_pending_human_review",template_family="visual_color_space_v3",
                     training_policy="experimental_machine_labels_only_not_human_accepted",contrast_set_id=None,paraphrase_group_id=None,
                     spatial_reference="image_view",
                     query_types=["combined" if spatial and appearance else "spatial" if spatial else "appearance" if appearance else "category"])
            rows.append(q)
        audit.append(dict(semantic_query_id=sid,image_id=a["image_id"],target_instance_id=aid,action=action,
                          sheet=sheet[a["image_id"]],reason=reason,texts=texts,
                          uniqueness_basis="Codex visual judgment using full frame and target crop; focused candidate inspection where recorded. Pending human review."))
    replacements = {aid:(iid,zh,en) for iid,aid,zh,en in spec["replacements"]}
    require(len(replacements)==len(spec["replacements"]), "duplicate replacement target")
    for aid,(iid,zh,en) in replacements.items():
        require(aid in anns and anns[aid]["image_id"]==iid, "replacement target/image mismatch")
    used = set()
    for sid, pair in ps.items():
        z=pair["zh"]; aid=z["target_instance_id"]; d=overrides.get(str(aid),{})
        if aid in replacements:
            iid,zh,en=replacements[aid]; used.add(aid)
            emit(z,aid,dict(zh=zh,en=en),sid,"revised","Focused re-adjudication")
        elif d.get("action")=="quarantine":
            require(bool(d.get("reason")), "quarantine reason required")
            audit.append(dict(semantic_query_id=sid,image_id=z["image_id"],target_instance_id=aid,action="quarantine",reason=d["reason"],sheet=sheet[z["image_id"]]))
            for q in pair.values():
                quarantined.append(dict(q,quarantine_reason=d["reason"]))
        else:
            texts={lang:d.get(lang,pair[lang]["query"]) for lang in ("zh","en")}
            action="revised" if any(texts[l]!=pair[l]["query"] for l in texts) else "retained"
            emit(z,aid,texts,sid,action)
    for n,(aid,(iid,zh,en)) in enumerate(sorted(replacements.items()),1):
        if aid in used: continue
        base=next(q for q in old if q["image_id"]==iid)
        emit(base,aid,dict(zh=zh,en=en),version+"-r"+str(n).zfill(4),"replacement")
    pairs(rows)
    require(len({q["query_id"] for q in rows})==len(rows), "duplicate query ID")
    missing={i["image_id"] for i in manifest["images"]}-{q["image_id"] for q in rows}
    require(missing=={i["image_id"] for i in spec["unresolved_images"]}, "unrecorded missing image coverage")
    return rows,quarantined,audit

def build(parent, decisions, index_path, dev, out):
    require(not out.exists(), "output exists")
    spec=read_json(decisions)
    require(spec["parent_queries_sha256"]==sha256(parent/"queries_draft.jsonl"), "parent query mismatch")
    for rel,h in read_json(parent/"SHA256SUMS.json").items():
        require(sha256(parent/rel)==h, "parent checksum mismatch: "+rel)
    manifest=read_json(parent/"image_manifest.json"); coco=read_json(parent/"annotations_coco.json")
    old=[json.loads(l) for l in (parent/"queries_draft.jsonl").read_text().splitlines() if l.strip()]
    rows,quarantine,audit=prepare(manifest,coco,old,spec,read_json(index_path),out.name)
    dm=read_json(dev/"image_manifest.json")
    for field in ("image_id","group_id","sha256"):
        require(not {i[field] for i in manifest["images"]}&{i[field] for i in dm["images"]}, "train/dev overlap: "+field)
    require(not {(i["source_dataset"],i["source_image_id"]) for i in manifest["images"]}&{(i["source_dataset"],i["source_image_id"]) for i in dm["images"]}, "source overlap")
    out.mkdir(parents=True)
    for rel in ("images","masks"):
        shutil.copytree(parent/rel,out/rel)
    for rel in ("annotations_coco.json","mapping_config.json","source_subset.json"):
        shutil.copy2(parent/rel,out/rel)
    manifest["manifest_id"]=out.name;dump(out/"image_manifest.json",manifest)
    c=read_json(parent/"candidates_gt.json");c["candidate_set_id"]=out.name+"-gt";c["image_manifest"]["sha256"]=sha256(out/"image_manifest.json")
    dump(out/"candidates_gt.json",c)
    jsonl(out/"queries_draft.jsonl",rows);jsonl(out/"queries_machine_train.jsonl",rows);jsonl(out/"queries_accepted.jsonl",[])
    jsonl(out/"queries_quarantined.jsonl",quarantine);jsonl(out/"queries_parent.jsonl",old)
    dump(out/"visual_audit.json",audit);dump(out/"visual_decisions.json",spec);dump(out/"visual_index.json",read_json(index_path))
    notes=read_json(parent/"annotation_notes.json")
    dump(out/"annotation_notes.json",dict(disputes=notes.get("disputes",[]),unresolved_images=spec["unresolved_images"],
         policy="Color and image-space location first; obvious common appearance only. All output labels remain machine drafts."))
    devrows=[json.loads(l) for l in (dev/"queries_draft.jsonl").read_text().splitlines() if l.strip()]
    cov=coverage(rows,devrows);cov["before"]=coverage(old,devrows)["features"];cov["dev_sha256"]=sha256(dev/"queries_draft.jsonl")
    dump(out/"training_text_coverage.json",cov)
    report=validate_package(out)
    report.update(images=len(manifest["images"]),query_images=len({q["image_id"] for q in rows}),semantic_groups=len(rows)//2,
                  query_records=len(rows),human_accepted=0,quarantined_groups=len(quarantine)//2,actions=dict(Counter(a["action"] for a in audit)),
                  source_masks_unchanged=sha256(parent/"annotations_coco.json")==sha256(out/"annotations_coco.json"),
                  parent_partitions_unchanged=manifest["images"]==read_json(parent/"image_manifest.json")["images"],
                  train_dev_overlap={"image_id":0,"group_id":0,"sha256":0,"source_id":0},
                  same_category_competition=sum(q["same_category_instance_count"]>1 for q in rows if q["language"]=="zh"),
                  query_types=dict(Counter("+".join(q["query_types"]) for q in rows if q["language"]=="zh")))
    require(c["images"]==read_json(parent/"candidates_gt.json")["images"],"candidate order/content changed")
    for rel in ("images","masks"):
        require(all(sha256(p)==sha256(parent/p.relative_to(out)) for p in (out/rel).rglob("*") if p.is_file()),"asset changed")
    dump(out/"validation_report.json",report)
    dump(out/"RELEASE.json",dict(release_id=out.name,parent=parent.name,parent_queries_sha256=spec["parent_queries_sha256"],
         formal_freeze=False,accepted_language_queries=0,usage="Experimental machine-label training and human review only; not human-accepted or formal evaluation.",
         dev_unchanged_sha256=sha256(dev/"queries_draft.jsonl"),statistics=report))
    return report

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for k in ("parent","decisions","index","dev","output"):p.add_argument("--"+k,type=Path,required=True)
    a=p.parse_args();print(json.dumps(build(a.parent,a.decisions,a.index,a.dev,a.output),ensure_ascii=False,indent=2))
