"""Evaluate existing frozen M1 and B1/B2 on reviewed dev, with isolated CPU caches."""
import argparse,json,sys,time,subprocess
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
import fashion_grounding
from fashion_grounding.data.schema import read_json,write_json,require,sha256

def infer_scores(head,visual,text,base):
    with torch.inference_mode():
        residual=head(torch.from_numpy(visual)[None,:,:],torch.from_numpy(text)).numpy()[0]
    scores=np.asarray(base+residual,dtype=np.float32)
    require(np.isfinite(scores).all(),"nonfinite M1 score")
    return scores

def run(a):
    torch.set_num_threads(a.threads);torch.set_num_interop_threads(1)
    grounding=Path(a.grounding_source)
    require(subprocess.check_output(["git","-C",str(grounding),"rev-parse","HEAD"],text=True).strip()=="103dcc546714dd7c7866b4259d96a9825d8090ef","unvalidated grounding version")
    fashion_grounding.__path__.append(str(grounding/"src/fashion_grounding"))
    from fashion_grounding.grounding.inputs import CandidatePack,crop_regions
    from fashion_grounding.grounding.cache import FeatureCache,visual_spec,text_spec
    from fashion_grounding.grounding.predict import read_queries,rank_scores,prediction_record,stage_inputs
    from fashion_grounding.grounding.heads import AppearanceResidual
    from fashion_grounding.grounding.rules import position_scores,shared_parse
    from fashion_grounding.encoders.siglip import SiglipEncoder
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    require(sha256(a.checkpoint)=="ed320169fafc0218ed99ce42cc003c2af7416853972c38db05960e856b888050","M1 checkpoint mismatch")
    state=torch.load(a.checkpoint,map_location="cpu")
    require(state["method"]=="M1" and state["epoch"]==5,"wrong fixed M1 checkpoint")
    head=AppearanceResidual(state["dimension"]);head.load_state_dict(state["model"],strict=True);head.eval().requires_grad_(False)
    encoder=SiglipEncoder(a.snapshot,a.revision,"cpu",a.batch_size)
    require({k:v for k,v in encoder.signature.items() if k!="device"}=={k:v for k,v in state["encoder"].items() if k!="device"},"encoder differs beyond CPU device")
    cache=FeatureCache(out/"features")
    lexicon=read_json(ROOT/"configs/rule_parser.json")
    metadata=dict(owner="data_eval",executed_by="wkq",status="running",device="cpu",threads=a.threads,
        checkpoint_sha256=sha256(a.checkpoint),query_sha256=sha256(a.queries),
        encoder=encoder.signature,training_encoder_device=state["encoder"]["device"],
        optimizer_updates=0,source_commit=subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip(),
        adapter_sha256=sha256(__file__),grounding_source_commit="103dcc546714dd7c7866b4259d96a9825d8090ef",
        parameter_policy="fixed epoch5 and B2 lambda1; no search/early stopping/label revision from predictions",
        segmentation_policy="reuse published u18000+0.3 predictions; no new segmentation forwards",
        command=[sys.executable,*sys.argv],results={})
    write_json(out/"run_metadata.json",metadata)
    for condition,cp in [("gt",a.gt_candidates),("predicted",a.predicted_candidates)]:
        pack=CandidatePack(cp);queries=read_queries(a.queries,pack)
        require(all(q.split=="dev" for q in queries),"dev-only")
        require(pack.package["candidate_source"]==condition,"candidate condition mismatch")
        path=out/condition;path.mkdir();ref=stage_inputs(pack,path)
        unique={(q.query,q.language):q for q in queries}
        missing=[q for q in unique.values() if cache.get(text_spec(q.query,q.language,encoder.signature),1,encoder.dimension) is None]
        for k in range(0,len(missing),a.batch_size):
            group=missing[k:k+a.batch_size];vectors=encoder.encode_texts([q.query for q in group])
            for q,v in zip(group,vectors):cache.put(text_spec(q.query,q.language,encoder.signature),v[None,:])
        print(json.dumps(dict(stage=condition,texts=len(missing),seconds=time.monotonic()-start)),flush=True)
        ids=list(dict.fromkeys(q.image_id for q in queries));preds={m:[] for m in ("B1","B2","M1")};score_rows=[]
        for n,iid in enumerate(ids,1):
            image,cs=pack.load(iid)
            row=pack.rows[iid];visual=None
            if row["status"]=="ok":
                spec=visual_spec(pack,iid,encoder.signature)
                visual=cache.get(spec,len(cs),encoder.dimension)
                if visual is None:
                    crops=crop_regions(image,cs);visual=encoder.encode_images(crops)
                    for crop in crops:crop.close()
                    cache.put(spec,visual)
            if image is not None:image.close()
            for q in (q for q in queries if q.image_id==iid):
                scores={m:None for m in preds}
                if visual is not None:
                    text=cache.get(text_spec(q.query,q.language,encoder.signature),1,encoder.dimension)
                    base=rank_scores(visual,text,encoder.logit_scale,encoder.logit_bias)
                    scores={"B1":base,"B2":base+position_scores(cs,shared_parse(q.query,q.language,lexicon)),
                            "M1":infer_scores(head,visual,text,base)}
                for method in preds:
                    preds[method].append(prediction_record(q,pack,"wkq-dev-"+condition+"-"+method+"-20261117-r01",method,ref,cs,scores[method]))
                score_rows.append(dict(query_id=q.query_id,candidate_ids=[c.candidate_id for c in cs],scores={m:None if v is None else v.tolist() for m,v in scores.items()}))
            if n%25==0:print(json.dumps(dict(stage=condition,images=n,total=len(ids),seconds=time.monotonic()-start)),flush=True)
        for method,records in preds.items():
            (path/(method+"-predictions.jsonl")).write_text("".join(json.dumps(p,ensure_ascii=False,allow_nan=False)+"\n" for p in records))
        (path/"scores.jsonl").write_text("".join(json.dumps(p,ensure_ascii=False,allow_nan=False)+"\n" for p in score_rows))
        metadata["results"][condition]=dict(images=len(ids),queries=len(queries),candidate_sha256=pack.digest,
            predictions={m:sha256(path/(m+"-predictions.jsonl")) for m in preds},seconds=time.monotonic()-start)
        write_json(out/"run_metadata.json",metadata)
    metadata.update(status="completed",elapsed_seconds=time.monotonic()-start)
    write_json(out/"run_metadata.json",metadata)
    print(json.dumps(metadata["results"]),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for k in ("grounding-source","checkpoint","queries","gt-candidates","predicted-candidates","snapshot","revision","output"):p.add_argument("--"+k,required=True)
    p.add_argument("--threads",type=int,default=4);p.add_argument("--batch-size",type=int,default=16)
    run(p.parse_args())
