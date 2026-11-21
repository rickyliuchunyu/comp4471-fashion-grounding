"""Unified evaluator for all fixed methods; no filtering on output success."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from evaluate_predictions import evaluate_files
from fashion_grounding.data.schema import write_json

def run(root,package):
    summary={}
    for condition in ("gt","predicted"):
        folder=root/condition
        candidates=folder/"inputs"/("candidates_gt.json" if condition=="gt" else "candidates_predicted.json")
        for method in ("B1","B2","M1"):
            output=folder/(method+"-evaluation.json")
            result=evaluate_files(package/"queries_accepted.jsonl",folder/(method+"-predictions.jsonl"),candidates,package/"annotations_coco.json",output)
            summary[condition+"-"+method]=result["summary"]
            print(json.dumps({condition+"-"+method:result["summary"]}),flush=True)
    write_json(root/"comparison.json",summary)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("run",type=Path);p.add_argument("package",type=Path);a=p.parse_args();run(a.run,a.package)
