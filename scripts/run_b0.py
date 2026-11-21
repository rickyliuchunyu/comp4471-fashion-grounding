"""Portable CPU B0 run: explicit inference allowlist; no annotation fields passed."""
import argparse,json,shutil,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from fashion_grounding.data.schema import read_json,write_json,sha256,safe_path,require,unique_index
from fashion_grounding.data.candidates import load_candidates
from fashion_grounding.baselines.category_position import select_target

def run(candidate_path,query_path,lexicon_path,output,run_id):
    package,manifest,_=load_candidates(candidate_path)
    require(package["candidate_source"] in ("gt","predicted"),"B0 requires internal candidate classes")
    queries=[json.loads(s) for s in query_path.read_text().splitlines() if s.strip()]
    require(bool(queries),"empty queries");unique_index(queries,"query_id")
    require(all(q["review_status"]=="accepted" for q in queries),"accepted queries required")
    lexicon=read_json(lexicon_path)
    require(not output.exists(),"run output exists");output.mkdir(parents=True)
    root=candidate_path.parent;dest=output/"inputs";dest.mkdir()
    paths={candidate_path.name,package["image_manifest"]["path"]}
    for im in manifest["images"]:
        # file_name is relative to the manifest directory
        mp=Path(package["image_manifest"]["path"]).parent
        paths.add(str(mp/im["file_name"]))
    for im in package["images"]:
        paths.update(c["mask_reference"] for c in im["candidates"])
    for rel in paths:
        src=safe_path(root,rel);dst=safe_path(dest,rel);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
    shutil.copyfile(query_path,output/"queries.jsonl");shutil.copyfile(lexicon_path,output/"lexicon.json")
    ims=unique_index(package["images"],"image_id");manifest_ims=unique_index(manifest["images"],"image_id")
    predictions=[]
    for q in queries:
        im=ims[q["image_id"]]
        require(q["split"]==manifest_ims[q["image_id"]]["split"],"query split mismatch")
        candidates=[{k:c[k] for k in ("candidate_id","category_id","bbox_xyxy")} for c in im["candidates"]]
        result=select_target(q["query"],q["language"],candidates,lexicon,im["status"])
        predictions.append({"schema_version":3,"illustrative_only":False,
            **{k:q[k] for k in ("query_id","image_id","query","language","split")},
            "run_id":run_id,"method_id":"B0","candidate_source":package["candidate_source"],
            "candidate_set_id":package["candidate_set_id"],
            "candidate_manifest":{"path":"inputs/"+candidate_path.name,"sha256":sha256(candidate_path)},
            **result,"error":{"code":"upstream_inference_error","message":"Candidate generation failed"} if result["status"]=="inference_error" else None})
    (output/"predictions.jsonl").write_text("".join(json.dumps(p,ensure_ascii=False)+"\n" for p in predictions),encoding="utf-8")
    write_json(output/"resolved_config.json",{"method_id":"B0","run_id":run_id,"lexicon_version":lexicon["lexicon_version"],"lexicon_sha256":sha256(lexicon_path),"query_sha256":sha256(query_path),"candidate_sha256":sha256(candidate_path),"selection":"unique class or unique box center extreme; reject unconsumed text","scope":"pilot diagnostic" if {q["split"] for q in queries}=={"train"} else "evaluation"})
    return predictions
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for key in ("candidates","queries","lexicon","output"):p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--run-id",required=True);a=p.parse_args()
    print(json.dumps({"predictions":len(run(a.candidates,a.queries,a.lexicon,a.output,a.run_id))}))
