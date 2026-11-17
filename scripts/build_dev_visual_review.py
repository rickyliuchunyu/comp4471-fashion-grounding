"""Build a versioned dev-only visual draft; never promote machine drafts to accepted."""
import argparse
from collections import Counter, defaultdict
import copy
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fashion_grounding.data.schema import read_json, sha256, validate_package, require


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def jsonl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def prepare(parent, old_rows, specs, decisions, version):
    """Validate all inputs before writing; all old bilingual groups need an audit."""
    manifest = read_json(parent / "image_manifest.json")
    images = {r["image_id"]: r for r in manifest["images"]}
    require(all(r["split"] == "dev" for r in images.values()), "dev-only source required")
    annotations = read_json(parent / "annotations_coco.json")["annotations"]
    anns = {r["id"]: r for r in annotations}
    by_image = defaultdict(list)
    for a in annotations:
        if not a["iscrowd"]:
            by_image[a["image_id"]].append(a)
    old = defaultdict(dict)
    for q in old_rows:
        require(q["split"] == "dev" and q["image_id"] in images, "old query outside dev")
        require(q["language"] not in old[q["semantic_query_id"]], "duplicate old language")
        old[q["semantic_query_id"]][q["language"]] = q
    require(set(decisions) == set(old), "every old semantic group needs exactly one audit decision")
    seen = set()
    visual = {}
    for s in specs:
        key = (s["image"] + 10**9, s["target"] + 10**9)
        require(key not in seen, "duplicate visual target")
        require(key[0] in images and key[1] in anns, "unknown visual target")
        require(anns[key[1]]["image_id"] == key[0] and not anns[key[1]]["iscrowd"], "target/image mismatch")
        require(all(isinstance(s.get(l), str) and s[l].strip() for l in ("zh", "en")), "bilingual text required")
        seen.add(key)
        visual[key] = s
    require({k[0] for k in seen} == set(images), "visual coverage incomplete")
    rows, audit, quarantine = [], [], []
    used = set()
    def emit(sid, image_id, target_id, texts, query_types, original=None):
        a = anns[target_id]
        require(a["image_id"] == image_id and not a["iscrowd"], "invalid target")
        for lang in ("zh", "en"):
            q = copy.deepcopy(original[lang]) if original else {}
            q.update(schema_version=3, illustrative_only=False,
                     query_id=sid+"-"+lang, semantic_query_id=sid, image_id=image_id,
                     source_dataset="fashionpedia", split="dev", query=texts[lang], language=lang,
                     target_instance_id=target_id, target_category_id=a["category_id"],
                     target_source_category_id=a["source_category_id"],
                     target_source_category_name=a["source_category_name"],
                     same_category_instance_count=sum(x["category_id"]==a["category_id"] for x in by_image[image_id]),
                     same_source_category_instance_count=sum(x["source_category_id"]==a["source_category_id"] for x in by_image[image_id]),
                     query_types=query_types, spatial_reference="image_view",
                     annotation_version=version, annotator="codex_visual_author_20261117",
                     reviewer=None, review_status="needs_revision",
                     authorship="machine_authored_after_visual_inspection_pending_human_review",
                     template_family=None, paraphrase_group_id=None,
                     training_policy="dev_review_only_not_training_or_formal_evaluation",
                     visual_evidence_sheet=visual.get((image_id,target_id), {}).get("sheet"))
            # Old contrast sets describe the previous text; do not silently claim a validated new pair.
            q["contrast_set_id"] = None
            rows.append(q)
    for sid, pair in old.items():
        require(set(pair)=={"zh","en"}, "old pair incomplete")
        z = pair["zh"]
        require(pair["en"]["target_instance_id"]==z["target_instance_id"] and pair["en"]["image_id"]==z["image_id"], "old translation target mismatch")
        key=(z["image_id"],z["target_instance_id"])
        d=decisions[sid]
        require(d["action"] in ("retain","revise","quarantine") and d.get("reason") and d.get("sheet"), "invalid audit decision")
        entry={"semantic_query_id":sid,"image_id":key[0],"target_instance_id":key[1],
               "old_records":[pair["zh"],pair["en"]],"action":d["action"],"reason":d["reason"],
               "evidence_sheet":d["sheet"],"reviewer":"codex_visual_audit_20261117","human_accepted":False}
        if d["action"]=="quarantine":
            require(key not in visual, "quarantined target cannot be reintroduced as a visual draft")
            quarantine.extend(pair.values())
        else:
            texts=d.get("texts") or {l:pair[l]["query"] for l in ("zh","en")}
            require(d["action"]!="retain" or all(texts[l]==pair[l]["query"] for l in ("zh","en")), "retain changed text")
            emit(sid,*key,texts,d.get("query_types",z["query_types"]),original=pair)
            entry["new_texts"]=texts
            used.add(key)
        audit.append(entry)
    for n,(key,s) in enumerate(visual.items(),1):
        if key not in used:
            emit(version+f"-v{n:04d}",*key,s,s.get("query_types",["appearance"]))
    require({q["image_id"] for q in rows}==set(images), "draft coverage incomplete")
    return manifest, rows, audit, quarantine


def build(parent, old_package, specs_file, decisions_file, out):
    require(not out.exists(), "output exists; use a new version")
    old_rows=[json.loads(l) for l in (old_package/"queries_draft.jsonl").read_text().splitlines() if l.strip()]
    specs=read_json(specs_file); decisions=read_json(decisions_file)
    manifest,rows,audit,quarantine=prepare(parent,old_rows,specs,decisions,out.name)
    # Check immutable source release before copying, and retain the evidence.
    source_checks={}
    for source in (parent,old_package):
        checks=read_json(source/"SHA256SUMS.json")
        require(all(sha256(source/p)==h for p,h in checks.items()), "source release checksum mismatch")
        source_checks[source.name]={"files_verified":len(checks),"checksums_sha256":sha256(source/"SHA256SUMS.json")}
    shutil.copytree(parent,out)
    for stale in ("SHA256SUMS.json","RELEASE.json","validation_report.json","consumer_check_report.json"):
        (out/stale).unlink(missing_ok=True)
    manifest["manifest_id"]=out.name
    dump(out/"image_manifest.json",manifest)
    c=read_json(out/"candidates_gt.json")
    c["image_manifest"]["sha256"]=sha256(out/"image_manifest.json")
    # Candidate set identity stays fixed: masks/order/IDs remain the identical GT set.
    dump(out/"candidates_gt.json",c)
    jsonl(out/"queries_draft.jsonl",rows)
    jsonl(out/"queries_accepted.jsonl",[])
    jsonl(out/"queries_quarantined.jsonl",quarantine)
    dump(out/"old290_visual_audit.json",audit)
    shutil.copy2(specs_file,out/"visual_annotation_specs.json")
    shutil.copy2(decisions_file,out/"old_query_decisions.json")
    dump(out/"annotation_notes.json",{
      "authorship":"Codex individually inspected original/candidate-mask contact sheets; not human acceptance.",
      "purpose":"dev only; never feed these queries into training",
      "disputes":[],
      "quarantined":[{"semantic_query_id":a["semantic_query_id"],"reason":a["reason"]} for a in audit if a["action"]=="quarantine"],
      "source_mask_attention":{"1000010012":[1000006228,1000006231],"1000017656":[1000002198,1000002199],"1000012010":[1000007896,1000007897],"1000017959":[1000003981,1000003982]},
      "limitations":"Review full image for unannotated distractors. Tiny/occluded targets, source mask fragments and bilingual phrasing still need human review. Contrast groups are not yet human validated; prior contrast IDs are retained only in the audit."
    })
    report=validate_package(out)
    require(read_json(out/"annotations_coco.json")==read_json(parent/"annotations_coco.json"),"annotations changed")
    require(manifest["images"]==read_json(parent/"image_manifest.json")["images"],"partition changed")
    require(c["images"]==read_json(parent/"candidates_gt.json")["images"],"candidates changed")
    for folder in ("images","masks"):
        require(all(sha256(p)==sha256(parent/p.relative_to(out)) for p in (out/folder).iterdir()),"asset changed")
    old_images={q["image_id"] for q in old_rows}
    report.update(semantic_groups=len(rows)//2, old_records_audited=len(old_rows),
                  old_semantic_actions=dict(Counter(a["action"] for a in audit)),
                  old_image_coverage=len(old_images),new_image_coverage=len({q["image_id"] for q in rows}-old_images),
                  groups=len({i["group_id"] for i in manifest["images"]}),
                  human_accepted=0, quarantined_records=len(quarantine),
                  target_categories=dict(Counter(q["target_category_id"] for q in rows if q["language"]=="zh")),
                  query_types=dict(Counter("+".join(q["query_types"]) for q in rows if q["language"]=="zh")),
                  unique_texts={l:len({q["query"] for q in rows if q["language"]==l}) for l in ("zh","en")},
                  same_category_competition=sum(q["same_category_instance_count"]>1 for q in rows if q["language"]=="zh"),
                  source_checks=source_checks, parent_assets_and_partitions_unchanged=True)
    dump(out/"validation_report.json",report)
    dump(out/"RELEASE.json",{"release_id":out.name,"formal_freeze":False,"accepted_language_queries":0,
      "parents":[parent.name,old_package.name],"usage":"Human-review dev visual draft only. No training or formal evaluation before acceptance.",
      "specs_sha256":sha256(specs_file),"decisions_sha256":sha256(decisions_file)})
    dump(out/"SHA256SUMS.json",{str(p.relative_to(out)):sha256(p) for p in sorted(out.rglob("*")) if p.is_file() and p.name!="SHA256SUMS.json"})
    return report

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for key in ("parent","old-package","specs","decisions","output"):
        p.add_argument("--"+key,type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(build(a.parent,a.old_package,a.specs,a.decisions,a.output),ensure_ascii=False,indent=2))
