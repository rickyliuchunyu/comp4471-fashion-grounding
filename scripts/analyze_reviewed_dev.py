"""Summarize fixed dev errors and paired image-bootstrap uncertainty."""
import argparse,json,sys
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from fashion_grounding.data.schema import read_json,write_json,require
from revise_dev_queries import FEATURES
import re

def bootstrap(rows,other=None,key="correct"):
    groups=defaultdict(list)
    other={r["query_id"]:r for r in other} if other is not None else None
    for row in rows:
        value=float(row[key])
        if other is not None:value-=float(other[row["query_id"]][key])
        groups[row["image_id"]].append(value)
    values=list(groups.values());sums=np.array([sum(v) for v in values]);counts=np.array([len(v) for v in values])
    rng=np.random.default_rng(42);samples=rng.integers(0,len(values),size=(1000,len(values)))
    means=sums[samples].sum(1)/counts[samples].sum(1)
    return dict(estimate=float(sums.sum()/counts.sum()),ci95=np.quantile(means,[.025,.975]).tolist(),replicates=1000,seed=42,unit="image with all bilingual queries",images=len(values))
def aggregate(rows):
    n=len(rows)
    return dict(queries=n,images=len({r["image_id"] for r in rows}),correct=sum(r["correct"] for r in rows),
                accuracy=sum(r["correct"] for r in rows)/n if n else None,
                success=sum(r["success_at_0_5"] for r in rows),success_at_0_5=sum(r["success_at_0_5"] for r in rows)/n if n else None)
def analyze(root,package,train):
    qs={q["query_id"]:q for q in (json.loads(l) for l in (package/"queries_accepted.jsonl").read_text().splitlines())}
    evaluations={c+"-"+m:read_json(root/c/(m+"-evaluation.json")) for c in ("gt","predicted") for m in ("B1","B2","M1")}
    gt=evaluations["gt-M1"]["rows"];pred=evaluations["predicted-M1"]["rows"]
    require({r["query_id"] for r in gt}==set(qs),"wrong denominator")
    buckets=Counter("geometric_success" if r["success_at_0_5"] else "target_not_geometrically_available" if not r["geometric_recalled"] else "available_but_selected_mask_failed" for r in pred)
    association=Counter("matched_and_selected" if r["correct"] else "no_same_class_unique_match" if not r["covered"] else "matched_candidate_available_but_not_selected" for r in pred)
    same={}
    for label,ev in evaluations.items():
        same[label]={name:aggregate([r for r in ev["rows"] if (r["same_category_instance_count"]>1)==hard]) for name,hard in (("same_category_multiple",True),("single_same_category",False))}
    trainrows=[json.loads(l) for l in train.read_text().splitlines() if l.strip()]
    trainpairs=defaultdict(list)
    for q in trainrows:trainpairs[q["semantic_query_id"]].append(q["query"])
    feature={}
    for name,pattern in FEATURES.items():
        training=sum(bool(re.search(pattern," ".join(texts),re.I)) for texts in trainpairs.values())
        ids={qid for qid,q in qs.items() if re.search(pattern,q["query"],re.I)}
        feature[name]=dict(training_semantic_mentions=training,gt_M1=aggregate([r for r in gt if r["query_id"] in ids]),predicted_M1=aggregate([r for r in pred if r["query_id"] in ids]))
    uncertainty={}
    for condition in ("gt","predicted"):
        a=evaluations[condition+"-M1"]["rows"];b=evaluations[condition+"-B1"]["rows"]
        uncertainty[condition+"_M1_accuracy"]=bootstrap(a)
        uncertainty[condition+"_M1_minus_B1_accuracy"]=bootstrap(a,b)
        uncertainty[condition+"_M1_success"]=bootstrap(a,key="success_at_0_5")
        uncertainty[condition+"_M1_minus_B1_success"]=bootstrap(a,b,key="success_at_0_5")
    cases=[];used=set()
    for reason,rows in (
        ("candidate_missing",[r for r in pred if not r["geometric_recalled"]]),
        ("candidate_present_selection_failure",[r for r in pred if r["geometric_recalled"] and not r["success_at_0_5"]]),
        ("GT_selection_failure",[r for r in gt if not r["correct"]])):
        count=0
        for row in rows:
            if row["image_id"] in used:continue
            cases.append(dict(row,reason=reason,query=qs[row["query_id"]]["query"]))
            used.add(row["image_id"]);count+=1
            if count==4:break
    report=dict(query_records=len(qs),semantic_groups=len(qs)//2,images=len({q["image_id"] for q in qs.values()}),
        geometric_failure_decomposition=dict(buckets),association_failure_decomposition=dict(association),
        same_category_breakdown=same,feature_coverage=feature,image_bootstrap=uncertainty,examples=cases,
        caveats=["Reviewed subset only; nine pending semantic groups excluded before predictions.",
                 "Single seed42 M1 trained on old machine_v2, not new visual_v3.",
                 "B2 lambda1 fixed diagnostic, not selected on this dev.",
                 "Segmentation u18000 and threshold0.3 previously selected using dev; this is development, not held-out final-test performance.",
                 "Bootstrap reflects image sampling, not training-seed uncertainty.",
                 "Text feature counts are correlational, not proof of why an example failed."])
    write_json(root/"error_analysis.json",report)
    print(json.dumps({k:v for k,v in report.items() if k in ("geometric_failure_decomposition","association_failure_decomposition","same_category_breakdown","image_bootstrap")},indent=2))
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for k in ("run","package","train"):p.add_argument("--"+k,type=Path,required=True)
    a=p.parse_args();analyze(a.run,a.package,a.train)
