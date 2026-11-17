"""Publish visually authored language increments; preserve base records and split boundaries."""
import argparse,copy,json,os,sys,re
from pathlib import Path
from collections import Counter,defaultdict
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from fashion_grounding.data.schema import read_json,write_json,sha256,require,validate_package
from build_dev_visual_review import jsonl
B=Path("/root/autodl-tmp/fashion-grounding");P=B/"shared/processed"
def validate_decisions(selected,decisions):
 require(set(decisions)=={str(x["index"]) for x in selected},"every selected image needs one decision")
 for key,d in decisions.items():
  require(d.get("reason"),"decision needs evidence/reason")
  if d.get("reject"):continue
  require(set(d["texts"])=={"zh","en"} and all(t.strip() for t in d["texts"].values()),"bilingual texts required")
  require(d["type"] in ("appearance","spatial","combined","category"),"invalid type")
  require(not re.search(r"手上|身上|穿着|拿着|流苏|亚麻|丝绸|真皮|held|wearing|linen|silk|leather", " ".join(d["texts"].values()),re.I),"unsupported fine/body relation")
def asset_copy(src,dst):
 dst.parent.mkdir(parents=True,exist_ok=True)
 if dst.exists():require(sha256(src)==sha256(dst),"asset collision")
 else:os.link(src,dst)
def build(source,selected,decisions,out,split,base=None):
 require(not out.exists(),"output exists")
 validate_decisions(selected,decisions)
 for root in [source]+([base] if base else []):
  require(all(sha256(root/k)==h for k,h in read_json(root/"SHA256SUMS.json").items()),"source checksum mismatch") if (root/"SHA256SUMS.json").exists() else None
 sm=read_json(source/"image_manifest.json");sc=read_json(source/"annotations_coco.json");ss=read_json(source/"candidates_gt.json")
 require(all(i["split"]==split for i in sm["images"]),"wrong source split")
 anns={a["id"]:a for a in sc["annotations"]};by=defaultdict(list)
 for a in anns.values():
  if not a.get("iscrowd"):by[a["image_id"]].append(a)
 keep={s["image_id"] for s in selected if not decisions[str(s["index"])].get("reject")}
 imap={i["image_id"]:i for i in sm["images"]};rows=[];audit=[]
 for s in selected:
  d=decisions[str(s["index"])];audit.append(dict(s,decision=d))
  if d.get("reject"):continue
  aid=d.get("target_id",s["target_id"]);a=anns[aid];im=imap[s["image_id"]]
  require(a["image_id"]==im["image_id"] and not a.get("iscrowd"),"invalid target")
  sid=out.name+f"-q{s['index']:04d}"
  for lang in ("zh","en"):
   rows.append(dict(schema_version=3,illustrative_only=False,query_id=sid+"-"+lang,semantic_query_id=sid,
    contrast_set_id=None,paraphrase_group_id=None,image_id=im["image_id"],source_dataset="fashionpedia",split=split,
    query=d["texts"][lang],language=lang,target_instance_id=aid,target_category_id=a["category_id"],
    same_category_instance_count=sum(x["category_id"]==a["category_id"] for x in by[im["image_id"]]),
    same_source_category_instance_count=sum(x["source_category_id"]==a["source_category_id"] for x in by[im["image_id"]]),
    target_source_category_id=a["source_category_id"],target_source_category_name=a["source_category_name"],
    query_types=[d["type"]],spatial_reference="image_view",annotation_version=out.name,
    annotator="codex_visual_author_20261117",reviewer=None,review_status="needs_revision",
    authorship="machine_authored_after_visual_inspection_pending_human_review",template_family="color_space_targeted_v1",
    visual_evidence_sheet=s["sheet"],training_policy="experimental_machine_labels_only_not_human_accepted" if split=="train" else "dev_review_only_not_training_or_formal_evaluation"))
 manifest=copy.deepcopy(sm);manifest["manifest_id"]=out.name;manifest["images"]=[i for i in sm["images"] if i["image_id"] in keep]
 coco=copy.deepcopy(sc);coco["images"]=[i for i in sc["images"] if i["id"] in keep];coco["annotations"]=[a for a in sc["annotations"] if a["image_id"] in keep]
 cs=copy.deepcopy(ss);cs["images"]=[i for i in ss["images"] if i["image_id"] in keep];cs["candidate_set_id"]=out.name+"-gt"
 oldrows=[]
 if base:
  bm=read_json(base/"image_manifest.json")
  for field in ("image_id","group_id","sha256"):
   require(not {i[field] for i in bm["images"]}&{i[field] for i in manifest["images"]},"base/new overlap: "+field)
  oldrows=[json.loads(l) for l in (base/"queries_draft.jsonl").read_text().splitlines() if l.strip()]
  manifest["images"]=bm["images"]+manifest["images"]
  bc=read_json(base/"annotations_coco.json");coco["images"]=bc["images"]+coco["images"];coco["annotations"]=bc["annotations"]+coco["annotations"]
  cs["images"]=read_json(base/"candidates_gt.json")["images"]+cs["images"]
 # Cross-partition guard independently from the source deduplication audit.
 protections=["fp_dev250_v1","fp_test400_holdout_v1"] if split=="train" else ["fp_train10000_targeted_v1","fp_dev250_v1","fp_test400_holdout_v1"]
 overlaps={}
 for name in protections:
  mm=read_json(P/name/"image_manifest.json")
  for field in ("image_id","group_id","sha256"):
   n=len({i[field] for i in manifest["images"]}&{i[field] for i in mm["images"]});overlaps[name+":"+field]=n
   require(n==0,"split overlap")
 out.mkdir(parents=True)
 roots=[(source,keep)]
 if base:roots.append((base,{i["image_id"] for i in read_json(base/"image_manifest.json")["images"]}))
 for root,ids in roots:
  for im in read_json(root/"image_manifest.json")["images"]:
   if im["image_id"] in ids:asset_copy(root/im["file_name"],out/im["file_name"])
  for im in read_json(root/"candidates_gt.json")["images"]:
   if im["image_id"] in ids:
    for c in im["candidates"]:asset_copy(root/c["mask_reference"],out/c["mask_reference"])
 write_json(out/"image_manifest.json",manifest);write_json(out/"annotations_coco.json",coco)
 cs["image_manifest"]["sha256"]=sha256(out/"image_manifest.json");cs["provenance"]["annotation_manifest_sha256"]=sha256(out/"annotations_coco.json")
 write_json(out/"candidates_gt.json",cs)
 write_json(out/"mapping_config.json",read_json(source/"mapping_config.json"))
 source_ids={int(i["source_image_id"]) for i in manifest["images"]}
 raw=read_json(source/"source_subset.json")
 require(source_ids<={i["id"] for i in raw["images"]},"source subset incomplete")
 raw["images"]=[i for i in raw["images"] if i["id"] in source_ids]
 raw["annotations"]=[a for a in raw["annotations"] if a["image_id"] in source_ids]
 write_json(out/"source_subset.json",raw)
 allrows=oldrows+rows
 jsonl(out/"queries_draft.jsonl",allrows);jsonl(out/"queries_machine_train.jsonl",allrows if split=="train" else [])
 jsonl(out/"queries_accepted.jsonl",[]);jsonl(out/"queries_increment.jsonl",rows)
 write_json(out/"visual_decisions.json",audit);write_json(out/"annotation_notes.json",dict(disputes=[],human_review="pending",policy="Simple colors and image-space location; common appearance only; source masks unchanged."))
 report=validate_package(out)
 report.update(images=len(manifest["images"]),query_images=len({q["image_id"] for q in allrows}),increment_images=len(keep),
  increment_semantic_groups=len(rows)//2,semantic_groups=len(allrows)//2,query_records=len(allrows),accepted=0,
  rejected_images=len(selected)-len(keep),split_overlap=overlaps,base_query_records_preserved=len(oldrows),
  increment_types=dict(Counter(q["query_types"][0] for q in rows if q["language"]=="zh")),
  increment_categories=dict(Counter(q["target_category_id"] for q in rows if q["language"]=="zh")),
  increment_same_category_groups=sum(q["same_category_instance_count"]>1 for q in rows if q["language"]=="zh"),
  increment_same_fine_groups=sum(q["same_source_category_instance_count"]>1 for q in rows if q["language"]=="zh"))
 write_json(out/"validation_report.json",report)
 write_json(out/"RELEASE.json",dict(release_id=out.name,formal_freeze=False,split=split,source=str(source),base=str(base) if base else None,
  usage="Experimental weak-label training; human review pending" if split=="train" else "Separate targeted dev challenge; human review required before evaluation; never train",
  audit="Full image and cyan target mask contact sheet visually inspected by Codex; not independent human review",
  source_annotation_sha256=sha256(source/"annotations_coco.json"),base_queries_sha256=sha256(base/"queries_draft.jsonl") if base else None,statistics=report))
 write_json(out/"SHA256SUMS.json",{str(p.relative_to(out)):sha256(p) for p in sorted(out.rglob("*")) if p.is_file() and p.name!="SHA256SUMS.json"})
 return report
if __name__=="__main__":
 p=argparse.ArgumentParser()
 for k in ("source","selection","decisions","output"):p.add_argument("--"+k,type=Path,required=True)
 p.add_argument("--base",type=Path);p.add_argument("--split",choices=["train","dev"],required=True)
 a=p.parse_args();print(json.dumps(build(a.source,read_json(a.selection),read_json(a.decisions),a.output,a.split,a.base),ensure_ascii=False,indent=2))
