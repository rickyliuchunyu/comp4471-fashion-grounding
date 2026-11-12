"""Apply recorded Codex visual revisions to an unpublished weak-label package."""
import argparse,json,sys
from pathlib import Path
from collections import Counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from fashion_grounding.data.schema import read_json,write_json,require,validate_package
def apply(root,patch_path):
 patch=read_json(patch_path);manifest=read_json(root/"image_manifest.json")
 require(patch["manifest_id"]==manifest["manifest_id"],"wrong package")
 require(not (root/"visual_revisions.json").exists(),"already revised")
 require(not (root/"queries_accepted.jsonl").read_text().strip(),"must not edit human accepted package")
 text=(root/"queries_draft.jsonl").read_text();rows=[json.loads(x) for x in text.splitlines() if x.strip()]
 edits={(e["image_id"],e["target_instance_id"]):e for e in patch["revisions"]}
 require(len(edits)==len(patch["revisions"]),"duplicate revision")
 matched=Counter();changed=set()
 for q in rows:
  key=(q["image_id"],q["target_instance_id"])
  if key not in edits:continue
  e=edits[key];require(q["review_status"]=="needs_revision","review state changed")
  require(q["query"]==e["before"][q["language"]],"unexpected original text")
  q["query"]=e["after"][q["language"]];q["query_types"]=[e["query_type"]]
  q["annotator"]="codex_visual_revision";q["authorship"]="machine_generated_with_codex_visual_revision_not_human_accepted"
  q["template_family"]="codex_visual_revision_v1";q["visual_revision_note"]=e["reason"]
  matched[key]+=1;changed.add(q["semantic_query_id"])
 require(set(matched)==set(edits) and all(n==2 for n in matched.values()),"missing bilingual revision")
 train=(root/"queries_machine_train.jsonl").read_text()
 require(not train or train==text,"unexpected training selection")
 (root/"queries_before_visual_revision.jsonl").write_text(text,encoding="utf-8")
 updated="".join(json.dumps(q,ensure_ascii=False)+"\n" for q in rows)
 (root/"queries_draft.jsonl").write_text(updated,encoding="utf-8")
 if train:(root/"queries_machine_train.jsonl").write_text(updated,encoding="utf-8")
 write_json(root/"visual_revisions.json",patch)
 evidence=read_json(root/"annotation_evidence.json")
 for e in evidence:
  if e["semantic_query_id"] in changed:e["codex_visual_revision"]=edits[(e["image_id"],e["target_instance_id"])]
 write_json(root/"annotation_evidence.json",evidence)
 report=read_json(root/"annotation_report.json")
 report.update(types=dict(Counter(q["query_types"][0] for q in rows if q["language"]=="zh")),individually_visual_reviewed=len(changed),visually_revised_semantics=len(changed),visual_review_kind="Codex visual revision only; not independent human acceptance")
 write_json(root/"annotation_report.json",report)
 write_json(root/"validation_report.json",validate_package(root))
 print(json.dumps({"visually_revised_semantics":len(changed),"accepted":0}))
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("package",type=Path);p.add_argument("patch",type=Path);a=p.parse_args();apply(a.package,a.patch)
