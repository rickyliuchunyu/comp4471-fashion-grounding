"""Independent real-pilot controls and exact frozen-cache replay; no GPU."""
import argparse
import copy
import json
from pathlib import Path
import sys
import numpy as np
from fashion_grounding.data.schema import read_json,write_json,decode_mask,require
from fashion_grounding.data.candidates import load_candidates
from fashion_grounding.evaluation.metrics import evaluate
from fashion_grounding.grounding.inputs import CandidatePack
from fashion_grounding.grounding.predict import read_queries,execute
from fashion_grounding.grounding.cache import FeatureCache

def main():
    p=argparse.ArgumentParser()
    for key in ("package","runs","cache","lexicon","output"):p.add_argument("--"+key,type=Path,required=True)
    a=p.parse_args();require(not a.output.exists(),"report exists")
    package,manifest,masks=load_candidates(a.package/"candidates_gt.json")
    coco=read_json(a.package/"annotations_coco.json")
    images={im["image_id"]:im for im in manifest["images"]}
    gt={ann["id"]:decode_mask(ann["segmentation"],images[ann["image_id"]]["height"],images[ann["image_id"]]["width"]) for ann in coco["annotations"]}
    raw=[json.loads(l) for l in (a.package/"queries_accepted.jsonl").read_text().splitlines() if l.strip()]
    baseline=[json.loads(l) for l in (a.runs/"B1/predictions.jsonl").read_text().splitlines() if l.strip()]
    def ev(preds):return evaluate(raw,preds,package,masks,coco["annotations"],gt)
    by_image={im["image_id"]:im for im in package["images"]}
    perfect=copy.deepcopy(baseline)
    for q,pred in zip(raw,perfect):
        cid=next(c["candidate_id"] for c in by_image[q["image_id"]]["candidates"] if c["source_annotation_id"]==q["target_instance_id"])
        pred["selected_candidate"]={"candidate_id":cid,"grounding_score":0.}
    result=ev(perfect)["summary"]
    require(result["oracle_selection_accuracy"]==result["mean_target_mask_iou"]==1.,"perfect control")
    empty=copy.deepcopy(perfect)
    for pred in empty:pred.update(status="no_output",selected_candidate=None)
    zero=ev(empty)["summary"]
    require(zero["queries"]==len(raw) and zero["oracle_selection_accuracy"]==zero["mean_target_mask_iou"]==0.,"failure denominator")
    rejected=[]
    bad=copy.deepcopy(perfect);bad[0]["selected_candidate"]["candidate_id"]="missing-id"
    wrong_query=copy.deepcopy(perfect);wrong_query[0]["query"]="different query"
    for label,preds in [("missing_prediction",perfect[:-1]),("unknown_candidate",bad),("mismatched_query",wrong_query)]:
        try:ev(preds)
        except ValueError:rejected.append(label)
        else:raise AssertionError(label+" accepted")
    pack=CandidatePack(a.package/"candidates_gt.json");queries=read_queries(a.package/"queries_accepted.jsonl",pack)
    replay={}
    for method in ("B1","B2"):
        meta=read_json(a.runs/method/"run_metadata.json")
        config=read_json(a.runs/method/"resolved_config.json")
        class CachedOnly:
            signature=meta["encoder"];dimension=signature["dimension"]
            logit_scale=signature["logit_scale"];logit_bias=signature["logit_bias"]
            def encode_images(self,_):raise AssertionError("cache miss would re-encode image")
            def encode_texts(self,_):raise AssertionError("cache miss would re-encode text")
        preds=[json.loads(l) for l in (a.runs/method/"predictions.jsonl").read_text().splitlines() if l.strip()]
        rows=execute(pack,queries,CachedOnly(),FeatureCache(a.cache),meta["run_id"],method,
            preds[0]["candidate_manifest"],config["rule_weight"],read_json(a.lexicon))
        require(all(err is None for _,err in rows) and [r for r,_ in rows]==preds,"cache replay differs")
        replay[method]={"exact_predictions":len(rows),"model_calls":0,"evaluation":ev(preds)["summary"]}
    b0=read_json(a.runs/"B0/evaluation.json")["summary"]
    require(b0["oracle_selection_accuracy"]==12/34 and b0["status_counts"]=={"rule_unsupported":22,"success":12},"B0 published pilot does not reproduce")
    write_json(a.output,{"status":"passed","scope":"real training pilot controls, no generalization claim",
        "command":[sys.executable,*sys.argv],"perfect_control":result,"no_output_control":zero,
        "rejected_controls":rejected,"cache_replay":replay,"B0_reproduced":b0})
    print(json.dumps({"status":"passed","controls":5,"exact_cache_replays":replay}))
if __name__=="__main__":main()
