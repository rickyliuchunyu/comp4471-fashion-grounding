"""Bounded CPU controlled M1/M2 development experiment; test never enters this API."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time
from types import SimpleNamespace
import numpy as np
import torch
from fashion_grounding.data.schema import read_json,write_json,sha256,require,validate_queries,unique_index
from .inputs import CandidatePack,crop_regions
from .predict import read_queries,rank_scores,prediction_record,stage_inputs
from .cache import FeatureCache,visual_spec,text_spec
from .geometry import geometry_features
from .spatial import make_head,residual,score_features
from .rules import position_scores,shared_parse,rule_coverage
from .heads import padded_batch,ranking_loss

def provenance():
    return {"git_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "working_tree_dirty":bool(subprocess.check_output(["git","status","--porcelain"],text=True).strip()),
        "command":[sys.executable,"-m","fashion_grounding.grounding.controlled",*sys.argv[1:]],
        "torch":torch.__version__,"numpy":np.__version__,"device":"cpu","owner":"grounding","executed_by":"wyz"}

def manifest_separation(paths):
    manifests={k:read_json(Path(p)/"image_manifest.json")["images"] for k,p in paths.items()}
    checks=[]
    for train in ("clean1000","expanded1149"):
        for held in ("dev","challenge","test"):
            for key in ("image_id","sha256","group_id"):
                count=len({r[key] for r in manifests[train]} & {r[key] for r in manifests[held]})
                require(count==0,"training overlap: "+train+"/"+held+"/"+key)
                checks.append({"train":train,"held":held,"field":key,"overlap":count})
    base={r["image_id"]:r for r in manifests["clean1000"]}
    expanded={r["image_id"]:r for r in manifests["expanded1149"]}
    require(all(expanded.get(i)==row for i,row in base.items()),"original train images changed")
    return checks

class FeatureBuilder:
    def __init__(self,config,out):
        self.config=config;self.out=out;self.encoder=None
        self.signature=read_json(config["previous_inference"]+"/run_metadata.json")["encoder"]
        require(self.signature["device"]=="cpu","CPU reference required")
        self.cache=FeatureCache(out/"features")
        self.readonly=FeatureCache(Path(config["previous_inference"])/"features")
        self.hits=0;self.encoded_images=0;self.encoded_texts=0
    def get(self,spec,n):
        result=self.cache.get(spec,n,self.signature["dimension"])
        if result is None:
            result=self.readonly.get(spec,n,self.signature["dimension"])
            if result is not None:self.hits+=1;self.cache.put(spec,result)
        return result
    def model(self):
        if self.encoder is None:
            from fashion_grounding.encoders.siglip import SiglipEncoder
            self.encoder=SiglipEncoder(self.config["snapshot"],self.signature["revision"],"cpu",16)
            require(self.encoder.signature==self.signature,"encoder signature mismatch")
        return self.encoder
    def build(self,name,candidate_path,query_path,annotations_path,training=False,*,draft_dev=False):
        pack=CandidatePack(candidate_path);queries=read_queries(query_path,pack,draft=training or draft_dev)
        raw=[json.loads(l) for l in Path(query_path).read_text().splitlines() if l.strip()]
        expected="train" if training else "dev"
        require(all(q.split==expected for q in queries),"wrong split")
        if draft_dev:
            require(not training and pack.package["candidate_source"]=="gt","draft diagnosis requires GT dev candidates")
            require(all(q["review_status"]=="needs_revision" for q in raw),"draft dev requires explicit unreviewed labels")
        if training:
            require(Path(query_path).name=="queries_machine_train.jsonl","explicit weak training file required")
            require(pack.package["candidate_source"]=="gt","GT-only training")
            require(all(r["review_status"]=="needs_revision" and r["reviewer"] is None for r in raw),"unexpected weak supervision")
        coco=read_json(annotations_path)
        validate_queries(raw,pack.images,unique_index(coco["annotations"],"id"),accepted_only=not (training or draft_dev))
        if pack.package["candidate_source"]=="gt":
            require(pack.package["provenance"]["annotation_manifest_sha256"]==sha256(annotations_path),"GT annotations changed")
        missing=[]
        for q in {(q.query,q.language):q for q in queries}.values():
            spec=text_spec(q.query,q.language,self.signature)
            if self.get(spec,1) is None:missing.append(q)
        for start in range(0,len(missing),16):
            group=missing[start:start+16];vectors=self.model().encode_texts([q.query for q in group])
            for q,v in zip(group,vectors):self.cache.put(text_spec(q.query,q.language,self.signature),v[None])
            self.encoded_texts+=len(group)
        by_image={};ids=list(dict.fromkeys(q.image_id for q in queries))
        for index,iid in enumerate(ids,1):
            image,cs=pack.load(iid);geometry=geometry_features(cs,image.size)
            spec=visual_spec(pack,iid,self.signature)
            visual=self.get(spec,len(cs)) if cs else np.empty((0,self.signature["dimension"]),np.float32)
            if visual is None:
                crops=crop_regions(image,cs);visual=self.model().encode_images(crops)
                for crop in crops:crop.close()
                self.cache.put(spec,visual);self.encoded_images+=1
            image.close()
            compact=[SimpleNamespace(candidate_id=c.candidate_id,category_id=c.category_id,bbox_xyxy=c.bbox_xyxy) for c in cs]
            by_image[iid]=(visual,geometry,compact)
            if index%100==0:print(json.dumps({"stage":"features","dataset":name,"images":index,"total":len(ids)}),flush=True)
        examples=[]
        for q in queries:
            visual,geometry,cs=by_image[q.image_id]
            text=self.get(text_spec(q.query,q.language,self.signature),1)
            base=rank_scores(visual,text,self.signature["logit_scale"],self.signature["logit_bias"]) if cs else np.empty(0,np.float32)
            examples.append({"query":q,"visual":visual,"geometry":geometry,"candidates":cs,"text":text,"base":base})
        return {"pack":pack,"queries":queries,"raw":raw,"examples":examples,
            "query_path":str(query_path),"annotations_path":str(annotations_path),
            "hashes":{"candidates":pack.digest,"queries":sha256(query_path),"images":sha256(pack.manifest_path)}}

def supervised_samples(dataset):
    # Loss construction only. Inference examples never carry a target.
    samples=[]
    for e,q in zip(dataset["examples"],dataset["raw"]):
        rows=dataset["pack"].rows[e["query"].image_id]["candidates"]
        indices=[i for i,c in enumerate(rows) if c["source_annotation_id"]==q["target_instance_id"]]
        require(len(indices)==1,"training target missing or duplicated")
        samples.append({**e,"target":indices[0]})
    return samples

def training_order(samples,seed,updates,batch_size):
    eligible=[i for i,s in enumerate(samples) if len(s["visual"])>1]
    require(bool(eligible),"no eligible queries")
    generator=torch.Generator().manual_seed(seed);order=[]
    while len(order)<updates*batch_size:
        order.extend(eligible[i] for i in torch.randperm(len(eligible),generator=generator).tolist())
    return order[:updates*batch_size]

def make_batch(samples,indices):
    batch=padded_batch(samples,indices)
    geometry=torch.zeros(*batch[0].shape[:2],12)
    for b,i in enumerate(indices):geometry[b,:len(samples[i]["geometry"])]=torch.from_numpy(samples[i]["geometry"])
    return (*batch,geometry)

def training_scores(model,batch):
    return (batch[2]+residual(model,batch[0],batch[1],batch[5])).masked_fill(~batch[3],float("-inf"))

def predict_scores(model,dataset):
    model.eval()
    return [score_features(model,e["visual"],e["text"],e["geometry"],e["base"]) if e["candidates"] else None for e in dataset["examples"]]

def gt_selection(dataset,scores):
    # This exact GT association criterion is used only for dev checkpoint selection.
    require(dataset["pack"].package["candidate_source"]=="gt","GT selection required")
    correct=0
    for e,q,score in zip(dataset["examples"],dataset["raw"],scores):
        if score is not None:
            c=dataset["pack"].rows[e["query"].image_id]["candidates"][int(np.argmax(score))]
            correct+=c["source_annotation_id"]==q["target_instance_id"]
    return int(correct)

def fit(config,method,train,dev,out,signature):
    out.mkdir();seed=config["seed"];torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    model=make_head(method,signature["dimension"]);torch.manual_seed(seed+1000)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
    samples=supervised_samples(train)
    order=training_order(samples,seed,config["updates"],config["batch_size"])
    order_hash=hashlib.sha256(np.asarray(order,dtype=np.int64).tobytes()).hexdigest()
    best=-1;best_path=None;history=[];start=time.monotonic();loss_sum=0.
    initial=predict_scores(model,dev)
    require(all(np.array_equal(s,e["base"]) for s,e in zip(initial,dev["examples"]) if s is not None),"zero initialization differs from B1")
    for update in range(1,config["updates"]+1):
        ids=order[(update-1)*config["batch_size"]:update*config["batch_size"]]
        batch=make_batch(samples,ids);model.train();optimizer.zero_grad(set_to_none=True)
        loss=ranking_loss(training_scores(model,batch),batch)
        require(bool(torch.isfinite(loss)),"nonfinite training loss")
        loss.backward();optimizer.step();loss_sum+=loss.item()
        if update%config["evaluation_interval"]==0 or update==config["updates"]:
            scores=predict_scores(model,dev);correct=gt_selection(dev,scores)
            record={"update":update,"dev_correct":correct,"dev_queries":len(dev["queries"]),
                "mean_loss_since_last_eval":loss_sum/(update-(history[-1]["update"] if history else 0))}
            history.append(record);loss_sum=0.
            checkpoint=out/f"update_{update:04d}.pt"
            torch.save({"method":method,"dimension":signature["dimension"],"model":model.state_dict(),
                "optimizer":optimizer.state_dict(),"encoder":signature,"update":update,
                "torch_rng":torch.get_rng_state(),"numpy_rng":np.random.get_state(),"python_rng":random.getstate(),
                "scheduler":None,"scaler":None,"training_order":order,"training_order_sha256":order_hash,
                "seed":seed,"config":config,"train_inputs":train["hashes"],"dev_inputs":dev["hashes"]},checkpoint)
            if correct>best:best=correct;best_path=checkpoint
            write_json(out/"training_log.json",history)
            print(json.dumps({"method":method,"dataset":out.name,**record}),flush=True)
    state=torch.load(best_path,map_location="cpu")
    selected=make_head(method,signature["dimension"]);selected.load_state_dict(state["model"]);selected.eval()
    restored=make_head(method,signature["dimension"]);restored.load_state_dict(torch.load(best_path,map_location="cpu")["model"]);restored.eval()
    scores=predict_scores(selected,dev);replay=predict_scores(restored,dev)
    require(all(np.array_equal(a,b) for a,b in zip(scores,replay)),"checkpoint replay")
    info={**provenance(),"method":method,"seed":seed,"optimizer_updates":config["updates"],
        "batch_size":config["batch_size"],"training_query_presentations":len(order),
        "eligible_queries":sum(len(s["visual"])>1 for s in samples),
        "single_candidate_excluded":sum(len(s["visual"])==1 for s in samples),
        "selection":"maximum accepted GT dev accuracy; tie earliest update",
        "selected_update":state["update"],"selected_dev_correct":best,"dev_queries":len(dev["queries"]),
        "checkpoint":str(best_path),"checkpoint_sha256":sha256(best_path),"training_order_sha256":order_hash,
        "trainable_parameters":sum(p.numel() for p in model.parameters()),"encoder_parameters_in_optimizer":0,
        "train_inputs":train["hashes"],"dev_inputs":dev["hashes"],
        "checkpoint_replay_exact":True,"training_labels":"AI weak labels, not human accepted",
        "elapsed_seconds":time.monotonic()-start,"status":"completed"}
    write_json(out/"run_metadata.json",info)
    return selected,info

def publish_scores(dataset,method,label,scores,out,config):
    from scripts.evaluate_predictions import evaluate_files
    # scripts is a namespace package when the repository root is on sys.path.
    folder=out/dataset["pack"].package["candidate_source"]
    if not folder.exists():
        folder.mkdir();stage_inputs(dataset["pack"],folder)
    ref={"path":"inputs/"+dataset["pack"].path.name,"sha256":dataset["pack"].digest}
    rows=[prediction_record(e["query"],dataset["pack"],config["run_id"]+"-"+dataset["pack"].package["candidate_source"]+"-"+label,method,ref,e["candidates"],score)
          for e,score in zip(dataset["examples"],scores)]
    pred=folder/(label+"-predictions.jsonl")
    pred.write_text("".join(json.dumps(r,ensure_ascii=False,allow_nan=False)+"\n" for r in rows))
    (folder/(label+"-scores.jsonl")).write_text("".join(json.dumps({"query_id":e["query"].query_id,
        "candidate_ids":[c.candidate_id for c in e["candidates"]],"scores":None if s is None else s.tolist()})+"\n" for e,s in zip(dataset["examples"],scores)))
    result=evaluate_files(Path(dataset["query_path"]),pred,dataset["pack"].path,Path(dataset["annotations_path"]),folder/(label+"-evaluation.json"))
    return result

def run(config):
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    try:
        run_in_new_directory(config, output)
    except Exception as exc:
        path = output / "run_metadata.json"
        if path.exists():
            metadata = read_json(path)
            metadata.update(status="failed", error_type=type(exc).__name__, error=str(exc))
            write_json(path, metadata)
        raise

def run_in_new_directory(config, out):
    start=time.monotonic()
    require(config["weak_label_training"] is True,"explicit weak-label setting required")
    require(0<config["updates"]<=490 and config["batch_size"]==32,"controlled trial budget exceeded")
    require(config["evaluation_interval"]>0,"invalid evaluation interval")
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    write_json(out/"resolved_config.json",config)
    metadata={**provenance(),"status":"running","scope":"development comparison with weak-label training; not final test"}
    write_json(out/"run_metadata.json",metadata)
    checks=manifest_separation(config["packages"]);write_json(out/"partition_checks.json",checks)
    builder=FeatureBuilder(config,out);datasets={}
    for name in ("clean1000","expanded1149","dev"):
        root=Path(config["packages"][name]);training=name!="dev"
        datasets[name]=builder.build(name,root/"candidates_gt.json",
            root/("queries_machine_train.jsonl" if training else "queries_accepted.jsonl"),root/"annotations_coco.json",training)
    devroot=Path(config["packages"]["dev"])
    datasets["predicted"]=builder.build("predicted",config["predicted_candidates"],devroot/"queries_accepted.jsonl",devroot/"annotations_coco.json")
    write_json(out/"feature_metadata.json",{"encoder":builder.signature,"readonly_cache_hits":builder.hits,
        "encoded_image_entries":builder.encoded_images,"encoded_text_entries":builder.encoded_texts,
        "inputs":{k:d["hashes"] for k,d in datasets.items()}})
    lexicon=read_json("configs/rule_parser.json")
    require(sha256(config["old_checkpoint"])=="ed320169fafc0218ed99ce42cc003c2af7416853972c38db05960e856b888050","old checkpoint differs")
    state=torch.load(config["old_checkpoint"],map_location="cpu")
    require({k:v for k,v in state["encoder"].items() if k!="device"}=={k:v for k,v in builder.signature.items() if k!="device"},"old model incompatible encoder")
    old=make_head("M1",state["dimension"]);old.load_state_dict(state["model"]);old.eval()
    summary={};reproduction={}
    for name in ("dev","predicted"):
        ds=datasets[name];condition=ds["pack"].package["candidate_source"]
        methods={"B1":[e["base"] if e["candidates"] else None for e in ds["examples"]],
            "B2":[e["base"]+position_scores(e["candidates"],shared_parse(e["query"].query,e["query"].language,lexicon)) if e["candidates"] else None for e in ds["examples"]],
            "M1-old":predict_scores(old,ds)}
        for label,scores in methods.items():
            method="M1" if label=="M1-old" else label
            result=publish_scores(ds,method,label,scores,out,config);summary[condition+"-"+label]=result["summary"]
            previous=read_json(Path(config["previous_inference"])/condition/(method+"-evaluation.json"))
            require(result["summary"]==previous["summary"],"B reported evaluation does not reproduce: "+condition+"/"+label)
            previous_rows={json.loads(l)["query_id"]:json.loads(l) for l in (Path(config["previous_inference"])/condition/(method+"-predictions.jsonl")).read_text().splitlines() if l.strip()}
            for e,s in zip(ds["examples"],scores):
                prev=previous_rows[e["query"].query_id]["selected_candidate"]
                if s is not None:
                    index=int(np.argmax(s))
                    require(prev["candidate_id"]==e["candidates"][index].candidate_id and np.isclose(prev["grounding_score"],float(s[index]),rtol=1e-5,atol=1e-5),"previous prediction differs")
            reproduction[condition+"-"+label]={"summary_exact":True,"all_selected_ids_equal":True,"score_rtol":1e-5,"score_atol":1e-5}
        write_json(out/(condition+"-B2-coverage.json"),rule_coverage(ds["pack"],ds["queries"],lexicon,1))
    write_json(out/"previous_evaluation_reproduction.json",reproduction)
    models={}
    for name in ("clean1000","expanded1149"):
        for method in ("M1","M2"):
            label=method+"-"+name
            model,info=fit(config,method,datasets[name],datasets["dev"],out/label,builder.signature)
            models[label]=info
            for split in ("dev","predicted"):
                ds=datasets[split];condition=ds["pack"].package["candidate_source"]
                result=publish_scores(ds,method,label,predict_scores(model,ds),out,config)
                summary[condition+"-"+label]=result["summary"]
            write_json(out/"comparison.json",summary)
    for name in ("clean1000","expanded1149"):
        require(models["M1-"+name]["training_order_sha256"]==models["M2-"+name]["training_order_sha256"],"M1/M2 data order differs")
    metadata.update(status="completed",elapsed_seconds=time.monotonic()-start,models=models,
        training_query_count={n:len(datasets[n]["queries"]) for n in ("clean1000","expanded1149")},
        dev_queries=len(datasets["dev"]["queries"]),other_seeds_not_run=[43,44],
        test_used_for_models=False,challenge_scored=False)
    write_json(out/"run_metadata.json",metadata)
    print(json.dumps({"status":"completed","seconds":metadata["elapsed_seconds"],"comparison":summary}),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--config",required=True);a=p.parse_args()
    run(read_json(a.config))
