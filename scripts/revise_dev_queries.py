"""Apply visually reviewed dev-only revisions and audit explicit train-text coverage."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import shutil
from build_dev_visual_review import prepare, dump, jsonl
from fashion_grounding.data.schema import read_json, sha256, validate_package, require

# Audit vocabulary only: never used by B0/inference or to rewrite training labels.
FEATURES = {
 "beige": r"米色|象牙色|\bbeige\b|\bivory\b",
 "black": r"黑|\bblack\b", "white": r"白|\bwhite\b", "gray": r"灰|\bgr[ae]y\b",
 "blue": r"蓝|\bblue\b", "red": r"红|\bred\b", "green": r"绿|\bgreen\b",
 "pink": r"粉|\bpink\b", "purple": r"紫|\bpurple\b", "orange": r"橙|\borange\b",
 "yellow": r"黄|\byellow\b", "brown": r"棕|褐|\bbrown\b", "gold": r"金色|金框|\bgold\b",
 "silver": r"银|\bsilver\b", "light": r"浅|\blight(?:-colored)?\b",
 "dark": r"深|\bdark\b", "multicolor": r"多色|彩色|\bmulticolou?red\b",
 "left": r"左|\bleft\b", "right": r"右|\bright\b", "middle": r"中央|中间|\bmiddle\b",
 "lower": r"下方|\blower\b", "stripe": r"条纹|条的|\bstriped\b",
 "check": r"格纹|格子|\bchecked\b|\bplaid\b", "dots": r"波点|圆点|\bdots?\b|\bdotted\b",
 "floral": r"花卉|花朵|\bfloral\b", "pattern": r"花纹|图案|印花|\bpatterned\b|\bpatterns\b|\bprinted\b",
 "leopard": r"豹纹|\bleopard\b", "camouflage": r"迷彩|\bcamouflage\b",
 "denim": r"牛仔|\bdenim\b|\bjeans\b", "hooded": r"连帽|\bhooded\b",
 "long_sleeve": r"长袖|\blong-sleeved\b", "short_sleeve": r"短袖|\bshort-sleeved\b",
 "sleeveless": r"无袖|\bsleeveless\b", "strapless": r"无肩带|\bstrapless\b",
 "cropped": r"短上衣|短夹克|\bcropped\b"
}
DISALLOWED = r"流苏|蕾丝|镂空|亮片|门襟|饰条|徽章|手持|手里|手中|手腕|身上|肩上|脸上|内搭|脖子|额头|头发|模特|穿着|戴着|披着|拿着|\b(?:holding|held|wearing|worn|wrist|forehead|fringe|lace|sequin|lettering)\b"

def pairs(rows):
    result = defaultdict(dict)
    for q in rows:
        require(q["language"] not in result[q["semantic_query_id"]], "duplicate language")
        result[q["semantic_query_id"]][q["language"]] = q
    for p in result.values():
        require(set(p) == {"zh", "en"}, "incomplete bilingual group")
        require(all(p["zh"][k] == p["en"][k] for k in ("image_id","target_instance_id","split")), "translation mismatch")
    return result

def features(pair):
    text = " ".join(q["query"] for q in pair.values())
    return {k for k, pattern in FEATURES.items() if re.search(pattern,text,re.I)}

def coverage(train_rows, dev_rows):
    tp, dp = pairs(train_rows), pairs(dev_rows)
    require(all(q["split"]=="train" for q in train_rows), "training source contains other splits")
    require(all(q["split"]=="dev" for q in dev_rows), "dev source contains other splits")
    require(not {q["image_id"] for q in train_rows} & {q["image_id"] for q in dev_rows}, "train/dev image overlap")
    tf={k:features(p) for k,p in tp.items()}; df={k:features(p) for k,p in dp.items()}
    counts={}
    for feature in FEATURES:
        counts[feature]={}
        for name, ps, fs in (("train",tp,tf),("dev",dp,df)):
            selected=[p["zh"] for sid,p in ps.items() if feature in fs[sid]]
            counts[feature][name]={"semantic_groups":len(selected),"images":len({q["image_id"] for q in selected}),
              "by_category":dict(Counter(str(q["target_category_id"]) for q in selected))}
    attention=[]
    for sid,p in dp.items():
        cat=str(p["zh"]["target_category_id"])
        missing=[f for f in sorted(df[sid]) if counts[f]["train"]["semantic_groups"]==0]
        sparse=[f for f in sorted(df[sid]) if 0<counts[f]["train"]["semantic_groups"]<10]
        category_missing=[f for f in sorted(df[sid]) if counts[f]["train"]["by_category"].get(cat,0)==0]
        if missing or sparse or category_missing:
            attention.append({"semantic_query_id":sid,"missing_train_text_features":missing,
                              "sparse_train_text_features":sparse,"missing_in_target_category":category_missing})
    return {"train_records":len(train_rows),"train_semantic_groups":len(tp),"train_images":len({q["image_id"] for q in train_rows}),
      "dev_records":len(dev_rows),"dev_semantic_groups":len(dp),"features":counts,"attention":attention,
      "method":"Explicit bilingual text mentions, deduplicated by semantic_query_id; union of translations. Counts do not prove visual absence or model inability; 1-9 is a screening threshold, not a learning guarantee. Machine training labels are not human-accepted. Dev labels were not copied into training."}

def build(parent, decisions_file, train, out):
    require(not out.exists(), "output exists; choose a new version")
    spec=read_json(decisions_file)
    require(spec["parent_queries_sha256"]==sha256(parent/"queries_draft.jsonl"), "parent query hash mismatch")
    for rel,h in read_json(parent/"SHA256SUMS.json").items():
        require(sha256(parent/rel)==h, "parent checksum mismatch")
    old=[json.loads(l) for l in (parent/"queries_draft.jsonl").read_text().splitlines() if l.strip()]
    ps=pairs(old); decisions=spec["decisions"]
    require(set(decisions)==set(ps), "complete decision coverage required")
    visual=[]
    for sid,p in ps.items():
        d=decisions[sid]; z=p["zh"]
        if d["action"]=="quarantine":
            continue
        require(d["compatible_candidate_ids"]==[z["target_instance_id"]], "unique visual target required; quarantine ambiguities")
        texts=d.get("texts") or {l:p[l]["query"] for l in ("zh","en")}
        require(not any(re.search(DISALLOWED,t,re.I) for t in texts.values()), "fine/body-relation text remains")
        visual.append({"image":z["image_id"]-10**9,"target":z["target_instance_id"]-10**9,
                       "sheet":d["sheet"],**texts})
    manifest, rows, audit, newly_quarantined=prepare(parent,old,visual,decisions,out.name)
    for q in rows:
        q["annotator"]="codex_visual_simplifier_20261117"
    for a in audit:
        a["compatible_candidate_ids"]=decisions[a["semantic_query_id"]].get("compatible_candidate_ids",[])
        a["uniqueness_basis"]="Machine visual inspection of full frame and all GT candidate crops; pending independent human review."
    train_rows=[json.loads(l) for l in train.read_text().splitlines() if l.strip()]
    train_hash=sha256(train)
    report=coverage(train_rows,rows)
    tm=read_json(train.parent/"image_manifest.json")
    require(not {i["group_id"] for i in tm["images"] if i["split"]=="train"} & {i["group_id"] for i in manifest["images"]}, "train/dev group overlap")
    report.update(train_file=str(train),train_sha256=train_hash,train_dev_image_overlap=0,train_dev_group_overlap=0)
    shutil.copytree(parent,out)
    for stale in ("SHA256SUMS.json","RELEASE.json","validation_report.json","consumer_check_report.json"):
        (out/stale).unlink(missing_ok=True)
    manifest["manifest_id"]=out.name; dump(out/"image_manifest.json",manifest)
    c=read_json(out/"candidates_gt.json"); c["image_manifest"]["sha256"]=sha256(out/"image_manifest.json")
    dump(out/"candidates_gt.json",c)
    jsonl(out/"queries_draft.jsonl",rows); jsonl(out/"queries_accepted.jsonl",[])
    prior=[json.loads(l) for l in (parent/"queries_quarantined.jsonl").read_text().splitlines() if l.strip()]
    jsonl(out/"queries_quarantined.jsonl",prior+newly_quarantined)
    dump(out/"simplification_audit.json",audit); dump(out/"simplification_decisions.json",spec)
    report["dev_sha256"]=sha256(out/"queries_draft.jsonl"); dump(out/"training_text_coverage.json",report)
    notes=read_json(out/"annotation_notes.json")
    notes["quarantined"] += [{"semantic_query_id":a["semantic_query_id"],"reason":a["reason"]} for a in audit if a["action"]=="quarantine"]
    notes["simplification"]="Colors and image-space location first; clear common appearance retained; fine detail/body relations removed. Earlier specs and old290 audit are historical v3 provenance only; current text is queries_draft.jsonl."
    dump(out/"annotation_notes.json",notes)
    validation=validate_package(out)
    require(c["images"]==read_json(parent/"candidates_gt.json")["images"],"candidate set changed")
    require(manifest["images"]==read_json(parent/"image_manifest.json")["images"],"split changed")
    require(sha256(out/"annotations_coco.json")==sha256(parent/"annotations_coco.json"),"masks changed")
    for folder in ("images","masks"):
        require(all(sha256(p)==sha256(parent/p.relative_to(out)) for p in (out/folder).rglob("*") if p.is_file()),"asset changed")
    require(sha256(train)==train_hash,"train changed")
    validation.update(semantic_groups=len(rows)//2,actions=dict(Counter(a["action"] for a in audit)),
      image_coverage=len({q["image_id"] for q in rows}),human_accepted=0,new_quarantined_records=len(newly_quarantined),
      total_quarantined_records=len(prior+newly_quarantined),parent_assets_and_partitions_unchanged=True,
      same_category_competition=sum(q["same_category_instance_count"]>1 for q in rows if q["language"]=="zh"),
      query_types=dict(Counter("+".join(q["query_types"]) for q in rows if q["language"]=="zh")))
    dump(out/"validation_report.json",validation)
    dump(out/"RELEASE.json",{"release_id":out.name,"formal_freeze":False,"accepted_language_queries":0,
      "parent":parent.name,"parent_queries_sha256":spec["parent_queries_sha256"],
      "usage":"Dev human-review draft only; no training or formal evaluation before acceptance.",
      "current_audit":"simplification_audit.json","historical_audit":"old290_visual_audit.json",
      "decisions_sha256":sha256(decisions_file),"training_unchanged_sha256":train_hash})
    dump(out/"SHA256SUMS.json",{str(p.relative_to(out)):sha256(p) for p in sorted(out.rglob("*")) if p.is_file() and p.name!="SHA256SUMS.json"})
    return validation

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for key in ("parent","decisions","train","output"):
        p.add_argument("--"+key,type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(build(a.parent,a.decisions,a.train,a.output),ensure_ascii=False,indent=2))
