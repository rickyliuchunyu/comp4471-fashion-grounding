"""Read-only, streaming grounding acceptance of published schema-3 GT packages."""
import argparse
from collections import Counter
import itertools
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import numpy as np
from fashion_grounding.data.schema import (read_json, write_json, sha256, safe_path,
    require, unique_index, validate_queries, decode_mask)
from fashion_grounding.grounding.inputs import CandidatePack, crop_regions
from fashion_grounding.grounding.predict import read_queries

def check_package(root):
    start=time.monotonic()
    checksums=read_json(root/"SHA256SUMS.json")
    for relative,digest in checksums.items():
        require(sha256(safe_path(root,relative))==digest,"release hash mismatch: "+relative)
    pack=CandidatePack(root/"candidates_gt.json")
    coco=read_json(root/"annotations_coco.json")
    require(sha256(root/"annotations_coco.json")==pack.package["provenance"]["annotation_manifest_sha256"],"annotation provenance")
    anns=unique_index(coco["annotations"],"id")
    ci=unique_index(coco["images"],"id")
    require(set(ci)==set(pack.images)==set(pack.rows),"image coverage")
    source=read_json(root/"source_subset.json")
    src=unique_index(source["annotations"],"id")
    mapping={r["source_id"]:r for r in read_json(root/"mapping_config.json")["fashionpedia"]}
    all_refs=set();cropped=0
    for iid,im in pack.images.items():
        require(all(ci[iid][k]==im[k] for k in ("width","height","file_name")),"COCO image metadata")
        offset=1 if im["source_split"]=="official_train2020" else 1000000001
        require(iid==int(im["source_image_id"])+offset,"image namespace")
        image,candidates=pack.load(iid)
        crops=crop_regions(image,candidates)
        cropped+=len(crops)
        for candidate,row in zip(candidates,pack.rows[iid]["candidates"]):
            aid=row["source_annotation_id"];a=anns[aid];original=src[a["source_annotation_id"]]
            require(aid not in all_refs,"duplicate GT reference");all_refs.add(aid)
            require(a["image_id"]==iid and not a["iscrowd"],"candidate GT association")
            require(aid==int(original["id"])+offset,"annotation namespace")
            require(original["image_id"]==int(im["source_image_id"]),"source association")
            rule=mapping[original["category_id"]]
            require(rule["action"]=="map" and candidate.category_id==a["category_id"]==rule["project_category_id"],"category mapping")
            require(a["source_category_id"]==original["category_id"],"source category")
            require(np.array_equal(candidate.mask,decode_mask(a["segmentation"],im["height"],im["width"])),"COCO mask differs")
            require(np.array_equal(candidate.mask,decode_mask(original["segmentation"],im["height"],im["width"])),"original source mask differs")
            x1,y1,x2,y2=candidate.bbox_xyxy
            require(a["area"]==int(candidate.mask.sum()) and a["bbox"]==[x1,y1,x2-x1,y2-y1],"COCO area/box")
        for crop in crops:crop.close()
        image.close()
    require(all_refs=={k for k,a in anns.items() if not a["iscrowd"]},"missing GT candidate")
    query_reports={}
    for filename in ("queries_draft.jsonl","queries_accepted.jsonl","queries_machine_train.jsonl"):
        path=root/filename
        if not path.exists():continue
        rows=[json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        validate_queries(rows,pack.images,anns,accepted_only=filename=="queries_accepted.jsonl")
        if rows:
            # Verify actual A inference input consumer, with drafts explicitly opted in.
            sanitized=read_queries(path,pack,draft=filename!="queries_accepted.jsonl")
            require(all(not hasattr(q,"target_instance_id") for q in sanitized),"target leaked into model input")
            paired={}
            for q in rows:paired.setdefault(q["semantic_query_id"],[]).append(q["language"])
            require(all(sorted(v)==["en","zh"] for v in paired.values()),"bilingual pairing")
        query_reports[filename]={"rows":len(rows),"semantic_groups":len({q["semantic_query_id"] for q in rows}),
            "review_status":dict(Counter(q["review_status"] for q in rows)),"sha256":sha256(path)}
    return {"status":"passed","images":len(pack.images),"instances":len(anns),"crops":cropped,
        "sha256_files":len(checksums),"image_manifest_sha256":sha256(root/"image_manifest.json"),
        "candidate_manifest_sha256":pack.digest,"queries":query_reports,"elapsed_seconds":time.monotonic()-start},pack.images

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--packages",nargs="+",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();require(not a.output.exists(),"output exists")
    report={"git_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "command":[sys.executable,*sys.argv],"python":platform.python_version(),"numpy":np.__version__,
        "scope":"technical consumer acceptance, not independent human semantic review",
        "packages":{},"cross_split":[],"subsets":{}}
    manifests={}
    for root in a.packages:
        result,images=check_package(root);report["packages"][root.name]=result;manifests[root.name]=images
        print(json.dumps({"package":root.name,**result}),flush=True)
    mains=["fp_train5000_v1","fp_dev250_v1","fp_test400_holdout_v1"]
    for left,right in itertools.combinations([n for n in mains if n in manifests],2):
        overlaps={k:len({x[k] for x in manifests[left].values()} & {x[k] for x in manifests[right].values()})
                  for k in ("image_id","sha256","group_id")}
        require(not any(overlaps.values()),"cross-split overlap")
        report["cross_split"].append({"left":left,"right":right,"overlaps":overlaps})
    for child,parent in [("fp_language1000_machine_v2","fp_train5000_v1"),("fp_language_dev_draft_v2","fp_dev250_v1")]:
        if child in manifests and parent in manifests:
            require(all(i in manifests[parent] and row==manifests[parent][i] for i,row in manifests[child].items()),"language subset metadata differs")
            report["subsets"][child]={"parent":parent,"exact_image_metadata":True}
    report["status"]="passed";write_json(a.output,report)
if __name__=="__main__":main()
