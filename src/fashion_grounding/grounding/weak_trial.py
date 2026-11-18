"""Explicit train-only weak-label M1 trial; no dev selection or formal accuracy."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import subprocess
import sys
import time
import numpy as np
import torch
from fashion_grounding.data.schema import read_json,write_json,sha256,require,validate_queries,unique_index
from .inputs import CandidatePack,crop_regions
from .cache import FeatureCache,visual_spec,text_spec
from .predict import read_queries,rank_scores,stage_inputs,prediction_record
from .heads import AppearanceResidual,padded_batch,batch_scores,ranking_loss

def context(args):
    pack=CandidatePack(args.candidates)
    require(pack.package["candidate_source"]=="gt","weak trial currently supports GT candidates only")
    queries=read_queries(args.queries,pack,draft=True)
    require(all(q.split=="train" for q in queries),"weak trial is train-only")
    require(Path(args.queries).name=="queries_machine_train.jsonl","explicit machine training manifest required")
    raw=[json.loads(line) for line in Path(args.queries).read_text().splitlines() if line.strip()]
    require(all(q["review_status"]=="needs_revision" and q["reviewer"] is None for q in raw),"unexpected weak-label review state")
    return pack,queries,raw

def metadata(args,pack):
    return {"scope":"weak_label_train_fit_not_generalization","diagnostic_only":True,
        "git_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "working_tree_dirty":bool(subprocess.check_output(["git","status","--porcelain"],text=True).strip()),
        "command":[sys.executable,"-m","fashion_grounding.grounding.weak_trial",*sys.argv[1:]],
        "candidate_manifest_sha256":pack.digest,"query_manifest_sha256":sha256(args.queries),
        "image_manifest_sha256":sha256(pack.manifest_path),"torch":torch.__version__,"numpy":np.__version__}

def prepare(args):
    pack,queries,_=context(args);out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    from fashion_grounding.encoders.siglip import SiglipEncoder
    encoder=SiglipEncoder(args.snapshot,args.revision,args.device,args.batch_size)
    cache=FeatureCache(args.cache);start=time.monotonic();image_count=0;text_count=0
    for index,iid in enumerate(dict.fromkeys(q.image_id for q in queries)):
        n=len(pack.rows[iid]["candidates"])
        require(n>0,"training image has no candidates")
        spec=visual_spec(pack,iid,encoder.signature)
        if cache.get(spec,n,encoder.dimension) is None:
            image,cs=pack.load(iid);crops=crop_regions(image,cs)
            cache.put(spec,encoder.encode_images(crops));image_count+=1
            for crop in crops:crop.close()
            image.close()
        if (index+1)%100==0:print(json.dumps({"images_cached":index+1,"seconds":time.monotonic()-start}),flush=True)
    unique={(q.query,q.language):q for q in queries}
    pending=[q for q in unique.values() if cache.get(text_spec(q.query,q.language,encoder.signature),1,encoder.dimension) is None]
    for offset in range(0,len(pending),args.batch_size):
        group=pending[offset:offset+args.batch_size]
        vectors=encoder.encode_texts([q.query for q in group])
        for q,vector in zip(group,vectors):
            cache.put(text_spec(q.query,q.language,encoder.signature),vector[None,:]);text_count+=1
    require(not encoder.model.training and not any(p.requires_grad for p in encoder.model.parameters()),"encoder not frozen")
    report={**metadata(args,pack),"encoder":encoder.signature,"queries":len(queries),
        "images":len({q.image_id for q in queries}),"new_visual_entries":image_count,"new_text_entries":text_count,
        "elapsed_seconds":time.monotonic()-start,"status":"completed"}
    write_json(out/"features.json",report);print(json.dumps({k:v for k,v in report.items() if k not in ("encoder","command")}),flush=True)

def load_samples(args,pack,queries,raw,signature):
    coco=read_json(Path(args.candidates).parent/"annotations_coco.json")
    require(sha256(Path(args.candidates).parent/"annotations_coco.json")==pack.package["provenance"]["annotation_manifest_sha256"],"training annotation hash")
    validate_queries(raw,pack.images,unique_index(coco["annotations"],"id"))
    cache=FeatureCache(args.cache);by_image={};samples=[];hard=0
    for q,row in zip(queries,raw):
        # Targets are used here only to construct supervised loss labels.
        cs=pack.rows[q.image_id]["candidates"]
        index=next((i for i,c in enumerate(cs) if c["source_annotation_id"]==row["target_instance_id"]),None)
        require(index is not None,"target missing from GT candidates")
        if q.image_id not in by_image:
            by_image[q.image_id]=cache.get(visual_spec(pack,q.image_id,signature),len(cs),signature["dimension"])
        visual=by_image[q.image_id]
        text=cache.get(text_spec(q.query,q.language,signature),1,signature["dimension"])
        require(visual is not None and text is not None,"missing frozen feature cache")
        category=cs[index]["category_id"]
        hard+=sum(c["category_id"]==category for c in cs)>1
        samples.append({"visual":visual,"text":text,
            "base":rank_scores(visual,text,signature["logit_scale"],signature["logit_bias"]),
            "target":index,"candidate_ids":[c["candidate_id"] for c in cs]})
    return samples,hard

def fit_epoch(model,optimizer,samples,batch_size):
    model.train();indices=[i for i,s in enumerate(samples) if len(s["visual"])>1]
    order=torch.randperm(len(indices)).tolist();loss_sum=0;count=0;updates=0
    for start in range(0,len(order),batch_size):
        ids=[indices[j] for j in order[start:start+batch_size]]
        batch=padded_batch(samples,ids)
        optimizer.zero_grad(set_to_none=True);loss=ranking_loss(batch_scores(model,batch),batch)
        require(bool(torch.isfinite(loss)),"nonfinite loss")
        loss.backward();optimizer.step()
        loss_sum+=loss.item()*len(ids);count+=len(ids);updates+=1
    require(count>0,"no eligible training examples")
    return {"mean_optimization_loss":loss_sum/count,"eligible_queries":count,"updates":updates}

def measure(model,samples,batch_size):
    model.eval();correct=0;total=0;loss_sum=0;eligible=0;all_scores=[]
    with torch.no_grad():
        for start in range(0,len(samples),batch_size):
            ids=list(range(start,min(start+batch_size,len(samples))))
            batch=padded_batch(samples,ids);scores=batch_scores(model,batch)
            correct+=int((scores.argmax(1)==batch[4]).sum());total+=len(ids)
            n=int((batch[3].sum(1)>1).sum())
            if n:loss_sum+=ranking_loss(scores,batch).item()*n;eligible+=n
            all_scores.extend(scores[i,:len(samples[j]["visual"])].numpy().copy() for i,j in enumerate(ids))
    return {"queries":total,"correct_against_weak_labels":correct,"weak_train_fit":correct/total,
            "multi_candidate_cross_entropy":loss_sum/eligible if eligible else None},all_scores

def checkpoint(model,optimizer,epoch,updates,signature,provenance):
    return {"model":model.state_dict(),"optimizer":optimizer.state_dict(),"epoch":epoch,"updates":updates,
        "torch_rng":torch.get_rng_state(),"numpy_rng":np.random.get_state(),"python_rng":random.getstate(),
        "dimension":model.dimension,"encoder":signature,"method":"M1","provenance":provenance,
        "scheduler":None,"scaler":None}

def restore(state,model,optimizer):
    model.load_state_dict(state["model"]);optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["torch_rng"]);np.random.set_state(state["numpy_rng"]);random.setstate(state["python_rng"])

def train(args):
    torch.set_num_threads(4)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    pack,queries,raw=context(args);features=read_json(args.features)
    require(features["candidate_manifest_sha256"]==pack.digest and features["query_manifest_sha256"]==sha256(args.queries),"cached training input changed")
    signature=features["encoder"];samples,hard=load_samples(args,pack,queries,raw,signature)
    model=AppearanceResidual(signature["dimension"])
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    provenance=metadata(args,pack);provenance["features_manifest_sha256"]=sha256(args.features)
    initial,_=measure(model,samples,args.batch_size)
    logs=[];updates=0;start=time.monotonic()
    for epoch in range(1,args.epochs+1):
        log={"epoch":epoch,**fit_epoch(model,optimizer,samples,args.batch_size)}
        updates+=log["updates"];log["elapsed_seconds"]=time.monotonic()-start
        logs.append(log);print(json.dumps(log),flush=True)
        torch.save(checkpoint(model,optimizer,epoch,updates,signature,provenance),out/f"epoch_{epoch:02d}.pt")
        write_json(out/"training_log.json",logs)
    final,scores=measure(model,samples,args.batch_size)
    final_path=out/f"epoch_{args.epochs:02d}.pt"
    # Restore into a new model and require identical inference before release.
    restored=AppearanceResidual(signature["dimension"]);restored.load_state_dict(torch.load(final_path,map_location="cpu")["model"])
    restored_result,restored_scores=measure(restored,samples,args.batch_size)
    require(final==restored_result and all(np.array_equal(a,b) for a,b in zip(scores,restored_scores)),"checkpoint replay differs")
    ref=stage_inputs(pack,out);predictions=[]
    from types import SimpleNamespace
    for q,s,score in zip(queries,samples,scores):
        candidates=[SimpleNamespace(candidate_id=cid) for cid in s["candidate_ids"]]
        predictions.append(prediction_record(q,pack,args.run_id,"M1",ref,candidates,score))
    (out/"weak_train_predictions.jsonl").write_text("".join(json.dumps(p,ensure_ascii=False)+"\n" for p in predictions))
    report={**provenance,"run_id":args.run_id,"status":"completed","method":"M1","seed":args.seed,
        "epochs":args.epochs,"batch_queries":args.batch_size,"learning_rate":3e-4,"weight_decay":.01,
        "selection":"fixed final epoch, no dev selection; no early stopping",
        "sampling":"uniform shuffled queries; no oversampling fabricated hard examples",
        "same_project_category_competition_queries":hard,"same_project_category_competition_fraction":hard/len(samples),
        "single_candidate_queries_excluded_from_loss":sum(len(s["visual"])==1 for s in samples),
        "trainable_parameters":sum(p.numel() for p in model.parameters()),"encoder_parameters_in_optimizer":0,
        "initial":initial,"final":final,"updates":updates,"elapsed_seconds":time.monotonic()-start,
        "checkpoint":{"path":final_path.name,"sha256":sha256(final_path)},
        "checkpoint_replay_exact":True,"formal_dev_test_results":None,"other_seeds_not_run":[43,44]}
    write_json(out/"run_metadata.json",report);write_json(out/"resolved_config.json",vars(args))
    print(json.dumps(report),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase",choices=["cache","train"],required=True)
    for key in ("candidates","queries","cache","output","run-id"):p.add_argument("--"+key,required=True)
    p.add_argument("--weak-label-trial",action="store_true",required=True)
    p.add_argument("--snapshot");p.add_argument("--revision");p.add_argument("--features")
    p.add_argument("--device",default="cuda");p.add_argument("--batch-size",type=int,default=32)
    p.add_argument("--epochs",type=int,default=5);p.add_argument("--seed",type=int,default=42)
    args=p.parse_args();require(0<args.epochs<=5 and args.batch_size>0,"trial budget must be 1..5 epochs")
    (prepare if args.phase=="cache" else train)(args)
if __name__=="__main__":main()
