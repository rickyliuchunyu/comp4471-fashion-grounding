"""B1/B2 predictions; explicit draft diagnostics, frozen caches and schema-3 outputs."""
import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
from fashion_grounding.data.schema import read_json, write_json, sha256, safe_path, require, unique_index
from .inputs import CandidatePack, inference_query, crop_regions
from .cache import FeatureCache, visual_spec, text_spec, check_features
from .rules import position_scores, shared_parse, rule_coverage

def rank_scores(visual,text,scale,bias):
    require(visual.ndim == 2 and text.shape == (1,visual.shape[1]), "feature dimensions differ")
    check_features(visual,*visual.shape); check_features(text,*text.shape)
    require(math.isfinite(scale) and math.isfinite(bias), "invalid logit calibration")
    with np.errstate(over="raise",invalid="raise"):
        scores=np.exp(scale)*(visual@text[0])+bias
    require(np.isfinite(scores).all(), "nonfinite ranking")
    return scores

def prediction_record(query,pack,run_id,method,candidate_ref,candidates,scores=None):
    require(query.image_id in pack.rows, "missing image candidates")
    require(query.split == pack.images[query.image_id]["split"], "query/image split mismatch")
    require(query.illustrative_only == pack.package["illustrative_only"], "mixed real/synthetic query")
    upstream=pack.rows[query.image_id]["status"]
    record={"schema_version":3,"illustrative_only":query.illustrative_only,
            **asdict(query),"run_id":run_id,"method_id":method,
            "candidate_source":pack.package["candidate_source"],
            "candidate_set_id":pack.package["candidate_set_id"],
            "candidate_manifest":candidate_ref,"status":upstream,
            "selected_candidate":None,"error":None}
    if upstream == "inference_error":
        record["error"]={"code":"upstream_inference_error","message":"Candidate producer failed; see its run record."}
    elif upstream == "ok":
        require(scores is not None and scores.shape == (len(candidates),) and np.isfinite(scores).all(), "invalid scores")
        best=int(np.argmax(scores))  # Stable producer order breaks equal grounding scores.
        record.update(status="success",selected_candidate={"candidate_id":candidates[best].candidate_id,
                                                          "grounding_score":float(scores[best])})
    return record

def read_queries(path,pack,draft=False,allow_test=False):
    rows=[json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    require(bool(rows), "empty query set")
    unique_index(rows,"query_id")
    queries=[]
    for row in rows:
        if not draft:
            require(row.get("review_status")=="accepted", "unreviewed queries require --draft-diagnostic")
            require(row.get("reviewer") and row.get("annotator") and row["reviewer"]!=row["annotator"], "independent review required")
        q=inference_query(row)
        require(q.image_id in pack.rows, "query image missing from candidate package")
        require(q.split==pack.images[q.image_id]["split"], "split mismatch")
        require(q.illustrative_only==pack.package["illustrative_only"], "mixed real/synthetic input")
        require(allow_test or q.split!="test", "final test requires --allow-final-test after team freeze")
        queries.append(q)
    require(len({q.split for q in queries})==1, "one split per run")
    return queries

def stage_inputs(pack,output):
    """Hard-link immutable shared bytes into a self-contained relative-path layout."""
    root=Path(output)/"inputs"
    root.mkdir()
    copied={}
    def copy(source,relative):
        dest=safe_path(root,str(relative))
        source=Path(source).resolve()
        if dest in copied:
            require(copied[dest]==sha256(source), "colliding input paths")
            return
        dest.parent.mkdir(parents=True,exist_ok=True)
        try: os.link(source,dest)
        except OSError: shutil.copy2(source,dest)
        copied[dest]=sha256(source)
    copy(pack.path,pack.path.name)
    ref=pack.package["image_manifest"]
    copy(pack.manifest_path,ref["path"])
    for im in pack.images.values():
        copy(safe_path(pack.manifest_path.parent,im["file_name"]),Path(ref["path"]).parent/im["file_name"])
    for row in pack.rows.values():
        for c in row["candidates"]:
            copy(safe_path(pack.path.parent,c["mask_reference"]),c["mask_reference"])
    return {"path":"inputs/"+pack.path.name,"sha256":pack.digest}

def execute(pack,queries,encoder,cache,run_id,method,ref,rule_weight=0,lexicon=None,parser=shared_parse):
    require(method in ("B1","B2"), "unknown baseline")
    require(math.isfinite(rule_weight) and rule_weight>=0, "invalid rule weight")
    rows=[]
    for q in queries:
        candidates=[]
        try:
            row=pack.rows[q.image_id]
            scores=None
            if row["status"]=="ok":
                image,candidates=pack.load(q.image_id)
                vs=visual_spec(pack,q.image_id,encoder.signature)
                visual=cache.get(vs,len(candidates),encoder.dimension)
                if visual is None:
                    visual=encoder.encode_images(crop_regions(image,candidates));cache.put(vs,visual)
                ts=text_spec(q.query,q.language,encoder.signature)
                text=cache.get(ts,1,encoder.dimension)
                if text is None:
                    text=encoder.encode_texts([q.query]);cache.put(ts,text)
                scores=rank_scores(visual,text,encoder.logit_scale,encoder.logit_bias)
                if method=="B2":
                    scores=scores+rule_weight*position_scores(candidates,parser(q.query,q.language,lexicon))
            record=prediction_record(q,pack,run_id,method,ref,candidates,scores)
        except (ValueError,RuntimeError,OSError) as exc:
            # Preserve the query denominator, with a sanitized public failure message.
            record={"schema_version":3,**asdict(q),"run_id":run_id,"method_id":method,
                    "candidate_source":pack.package["candidate_source"],
                    "candidate_set_id":pack.package["candidate_set_id"],
                    "candidate_manifest":ref,"status":"inference_error",
                    "selected_candidate":None,"error":{"code":type(exc).__name__,"message":"Grounding failed; inspect private run errors."}}
            rows.append((record,{"query_id":q.query_id,"type":type(exc).__name__,"message":str(exc)}))
            continue
        rows.append((record,None))
    return rows

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidates",required=True);p.add_argument("--queries",required=True)
    p.add_argument("--snapshot",required=True);p.add_argument("--revision",required=True)
    p.add_argument("--cache",required=True);p.add_argument("--output",required=True)
    p.add_argument("--run-id",required=True);p.add_argument("--method",choices=["B1","B2"],default="B1")
    p.add_argument("--device",default="cpu");p.add_argument("--batch-size",type=int,default=8)
    p.add_argument("--draft-diagnostic",action="store_true");p.add_argument("--allow-illustrative",action="store_true")
    p.add_argument("--allow-final-test",action="store_true")
    p.add_argument("--rule-weight",type=float,default=0)
    p.add_argument("--lexicon",default="configs/rule_parser.json")
    args=p.parse_args()
    pack=CandidatePack(args.candidates,args.allow_illustrative)
    queries=read_queries(args.queries,pack,args.draft_diagnostic,args.allow_final_test)
    lexicon=read_json(args.lexicon) if args.method=="B2" else None
    if lexicon is not None:
        # Missing shared parser is a setup failure, not 80 spurious query failures.
        shared_parse(queries[0].query,queries[0].language,lexicon)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    ref=stage_inputs(pack,output)
    from fashion_grounding.encoders.siglip import SiglipEncoder
    encoder=SiglipEncoder(args.snapshot,args.revision,args.device,args.batch_size)
    rows=execute(pack,queries,encoder,FeatureCache(args.cache),args.run_id,args.method,ref,args.rule_weight,lexicon)
    filename="draft_predictions.jsonl" if args.draft_diagnostic else "predictions.jsonl"
    with (output/filename).open("x",encoding="utf-8") as f:
        for record,_ in rows: f.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+"\n")
    errors=[error for _,error in rows if error]
    write_json(output/"errors.json",errors)
    write_json(output/"resolved_config.json",vars(args))
    write_json(output/"run_metadata.json",{
        "record_schema_version":1,"run_id":args.run_id,"experiment_id":args.method,
        "owner":"grounding","executed_by":"wyz",
        "status":"completed" if not errors else "failed",
        "diagnostic_only":args.draft_diagnostic or pack.package["illustrative_only"] or queries[0].split=="train",
        "rule_coverage":rule_coverage(pack,queries,lexicon,args.rule_weight) if lexicon else None,
        "git_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "split":queries[0].split,"candidate_source":pack.package["candidate_source"],
        "encoder":encoder.signature,"query_manifest_sha256":sha256(args.queries),
        "candidate_manifest_sha256":pack.digest,"query_count":len(rows),
        "status_counts":{s:sum(r["status"]==s for r,_ in rows) for s in {r["status"] for r,_ in rows}},
        "predictions":{"path":filename,"sha256":sha256(output/filename)},
        "lexicon_sha256":sha256(args.lexicon) if lexicon is not None else None,
        "command":[sys.executable,"-m","fashion_grounding.grounding.predict",*sys.argv[1:]],
        "working_tree_dirty":bool(subprocess.check_output(["git","status","--porcelain"],text=True).strip()),
        "source_sha256":{str(p):sha256(p) for folder in ("src/fashion_grounding/grounding","src/fashion_grounding/encoders","src/fashion_grounding/data") for p in sorted(Path(folder).glob("*.py"))},
        "evaluation":None,"note":"No accuracy or formal performance claimed. See handoff for input review status."})
    print(json.dumps({"output":str(output),"queries":len(rows),"errors":len(errors),"diagnostic_only":args.draft_diagnostic}))
    if errors: raise SystemExit(1)

if __name__=="__main__": main()
