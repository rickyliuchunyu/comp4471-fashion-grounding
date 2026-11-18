"""Select the prespecified B2 dev grid and summarize a controlled comparison."""
import argparse
from pathlib import Path
import json
from types import SimpleNamespace
import numpy as np
from fashion_grounding.data.schema import read_json,write_json,require,sha256
from fashion_grounding.grounding.inputs import CandidatePack
from fashion_grounding.grounding.predict import read_queries,prediction_record
from fashion_grounding.grounding.rules import position_scores,shared_parse,rule_coverage
from evaluate_predictions import evaluate_files
from analyze_reviewed_dev import bootstrap,aggregate

def choose_weight(results):
    best=max(r["correct"] for r in results)
    return min(r["weight"] for r in results if r["correct"]==best)

def select_b2(config,out):
    root=Path(config["packages"]["dev"]);lexicon=read_json("configs/rule_parser.json")
    prepared={}
    for condition in ("gt","predicted"):
        path=root/"candidates_gt.json" if condition=="gt" else Path(config["predicted_candidates"])
        pack=CandidatePack(path);queries=read_queries(root/"queries_accepted.jsonl",pack)
        cached={r["query_id"]:r for r in (json.loads(l) for l in (out/condition/"B1-scores.jsonl").read_text().splitlines() if l.strip())}
        require(set(cached)=={q.query_id for q in queries},"incomplete B1 score archive")
        entries=[]
        for q in queries:
            candidates=[SimpleNamespace(candidate_id=c["candidate_id"],category_id=c["category_id"],bbox_xyxy=c["bbox_xyxy"])
                        for c in pack.rows[q.image_id]["candidates"]]
            r=cached[q.query_id]
            require(r["candidate_ids"]==[c.candidate_id for c in candidates],"candidate order changed")
            base=None if r["scores"] is None else np.asarray(r["scores"],np.float32)
            if base is not None:require(base.shape==(len(candidates),) and np.isfinite(base).all(),"bad baseline scores")
            rule=position_scores(candidates,shared_parse(q.query,q.language,lexicon))
            entries.append((q,candidates,base,rule))
        prepared[condition]=(pack,entries)
    raw={q["query_id"]:q for q in (json.loads(l) for l in (root/"queries_accepted.jsonl").read_text().splitlines() if l.strip())}
    grid=[]
    pack,entries=prepared["gt"]
    for weight in (0,1,2,5,10):
        correct=0
        for q,cs,base,rule in entries:
            if base is not None:
                index=int(np.argmax(base+weight*rule))
                correct+=pack.rows[q.image_id]["candidates"][index]["source_annotation_id"]==raw[q.query_id]["target_instance_id"]
        grid.append({"weight":weight,"correct":int(correct),"queries":len(entries)})
    selected=choose_weight(grid);evaluations={}
    for condition,(pack,entries) in prepared.items():
        ref={"path":"inputs/"+pack.path.name,"sha256":pack.digest}
        records=[prediction_record(q,pack,config["run_id"]+"-"+condition+"-B2-selected","B2",ref,cs,
            None if base is None else base+selected*rule) for q,cs,base,rule in entries]
        path=out/condition/"B2-selected-predictions.jsonl"
        require(not path.exists(),"B2 selected output exists")
        path.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in records))
        evaluations[condition]=evaluate_files(root/"queries_accepted.jsonl",path,pack.path,
            root/"annotations_coco.json",out/condition/"B2-selected-evaluation.json")
    record={"selection_split":"dev","candidate_source_for_selection":"gt","grid":grid,
        "tie_break":"smallest weight","selected_weight":selected,
        "query_sha256":sha256(root/"queries_accepted.jsonl"),"lexicon_sha256":sha256("configs/rule_parser.json"),
        "candidate_sha256":prepared["gt"][0].digest,"fixed_for_predicted_candidates":True,
        "development_selection_not_final_test":True}
    write_json(out/"B2-dev-selection.json",record)
    return record

def analyze(config,out):
    labels=("B1","B2","B2-selected","M1-old","M1-clean1000","M2-clean1000","M1-expanded1149","M2-expanded1149")
    summaries={};subgroups={};evaluations={}
    for condition in ("gt","predicted"):
        for label in labels:
            d=read_json(out/condition/(label+"-evaluation.json"));evaluations[condition+"-"+label]=d
            summaries[condition+"-"+label]=d["summary"]
            rows=d["rows"]
            subgroups[condition+"-"+label]={}
            for name,rr in {
                "appearance":[r for r in rows if "appearance" in r["query_types"]],
                "combined":[r for r in rows if "combined" in r["query_types"]],
                "spatial":[r for r in rows if "spatial" in r["query_types"]],
                "same_category_multiple":[r for r in rows if r["same_category_instance_count"]>1],
                "same_category_single":[r for r in rows if r["same_category_instance_count"]==1]
            }.items():
                subgroups[condition+"-"+label][name]={**aggregate(rr),
                    "semantic_groups":len({r["semantic_query_id"] for r in rr})}
    paired={}
    for condition in ("gt","predicted"):
        for left,right in [
            ("M2-clean1000","M1-clean1000"),("M2-expanded1149","M1-expanded1149"),
            ("M1-expanded1149","M1-clean1000"),("M2-expanded1149","M2-clean1000"),
            ("M2-expanded1149","B2-selected")]:
            a=evaluations[condition+"-"+left]["rows"];b=evaluations[condition+"-"+right]["rows"]
            for metric in ("correct","success_at_0_5"):
                paired[condition+" "+left+" minus "+right+" "+metric]=bootstrap(a,b,key=metric)
    result={"summary":summaries,"subgroups":subgroups,"paired_image_bootstrap":paired,
        "caveats":["Single seed42, weak training labels; not a final-test claim.",
                   "Checkpoints and B2 weight selected on this dev; intervals do not correct selection bias.",
                   "Pure spatial subset contains only five bilingual semantic groups.",
                   "Historical M1-old is not a controlled old-vs-clean labeling ablation.",
                   "Challenge39 and final test400 were not scored."]}
    write_json(out/"controlled_analysis.json",result)
    print(json.dumps({"summary":summaries,"spatial_GT":{k:v["spatial"] for k,v in subgroups.items() if k.startswith("gt-")}},indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);a=p.parse_args()
    config=read_json(a.config);out=Path(config["output"])
    require(read_json(out/"run_metadata.json")["status"]=="completed","main experiment incomplete")
    select_b2(config,out);analyze(config,out)
