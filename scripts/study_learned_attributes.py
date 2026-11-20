"""Fit and test a supervised attribute ranker using only the locked MHP training split."""
import argparse
from collections import Counter
from itertools import product
import json
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from fashion_grounding.data.schema import read_json, require, sha256, write_json
from fashion_grounding.grounding.attributes import AttributeHeads, DIRECTIONS, add_components, ranking_components
from fashion_grounding.grounding.binding import binding_geometry
from fashion_grounding.grounding.controlled import predict_scores, publish_scores, supervised_samples
from fashion_grounding.grounding.inputs import CandidatePack
from fashion_grounding.grounding.rules import position_scores, shared_parse
from scripts.evaluate_direction_increment import COLOR
from scripts.evaluate_frozen_grounding_dev import check_release
from scripts.prepare_mhp_binding_features import build_inputs, builder_for, verify
from scripts.run_mhp_binding_study import pairs, pair_metrics, split_checks
from scripts.study_pair_checkpoint_selection import divide, model_at

SEED = 42
EPOCHS = 32


def direction_label(row, lexicon):
    if not {"spatial", "combined"} & set(row["query_types"]):
        return 0
    p = shared_parse(row["query"], row["language"], lexicon)
    if p.direction_status == "unique" and not p.unsupported_operator:
        return DIRECTIONS.index(p.direction)
    return -100


def fit_heads(train, lexicon):
    rows = train["raw"]
    text = torch.from_numpy(np.concatenate([e["text"] for e in train["examples"]]))
    targets_category = torch.tensor([r["target_category_id"]-1 for r in rows])
    targets_direction = torch.tensor([direction_label(r, lexicon) for r in rows])
    english = {q["semantic_query_id"]:q["query"] for q in rows if q["language"]=="en"}
    has_color = torch.tensor([float(bool(COLOR.search(english.get(q["semantic_query_id"], q["query"]))))
                              for q in rows])
    training_samples = supervised_samples(train)
    color_rows, differences = [], []
    for members in pairs(train).values():
        row = rows[members[0]]
        if row["source_dataset"] != "mhp_v2" or "appearance" not in row["query_types"]:
            continue
        for lang in ("en", "zh"):
            a,b = [i for i in members if rows[i]["language"]==lang]
            for i,j in ((a,b),(b,a)):
                features = train["examples"][i]["geometry"]
                correct, other = training_samples[i]["target"], training_samples[j]["target"]
                require(correct != other, "color pair must have distinct targets")
                color_rows.append(i)
                differences.append(features[correct,12:] - features[other,12:])
    require(len(color_rows) >= 284, "unexpected color supervision coverage")
    pair_text = text[torch.tensor(color_rows)]
    pair_diff = torch.from_numpy(np.stack(differences))
    model = AttributeHeads(text.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(),lr=.015,weight_decay=.01)
    rng = torch.Generator().manual_seed(SEED+1)
    losses = []
    for epoch in range(EPOCHS):
        step_loss = []
        order = torch.randperm(len(text),generator=rng)
        for idx in order.split(256):
            selected = text[idx]
            cidx = torch.randint(len(pair_text),(128,),generator=rng)
            direction_mask = targets_direction[idx] != -100
            loss = F.cross_entropy(model.direction(selected[direction_mask]),targets_direction[idx][direction_mask])
            loss = loss + F.cross_entropy(model.category(selected),targets_category[idx])
            loss = loss + F.binary_cross_entropy_with_logits(model.color_gate(selected).squeeze(-1),has_color[idx])
            learned_difference = (model.color_weights(pair_text[cidx]).tanh()*pair_diff[cidx]).sum(-1)
            loss = loss + F.softplus(1 - 5 * learned_difference).mean()
            require(bool(torch.isfinite(loss)),"nonfinite auxiliary loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            step_loss.append(float(loss.detach()))
        losses.append(sum(step_loss)/len(step_loss))
    return model.eval(), {"direction_labels":dict(Counter(targets_direction.tolist())),
                          "category_labels":dict(Counter(targets_category.tolist())),
                          "color_gate_positive":int(has_color.sum()),"color_pair_rows":len(color_rows),
                          "color_pair_groups":len(color_rows)//4,"epochs":EPOCHS,"epoch_mean_loss":losses}


def group_results(ds, values):
    selected = [None if v is None else int(v.argmax()) for v in values]
    targets = supervised_samples(ds)
    correct = [a == b["target"] for a,b in zip(selected, targets)]
    output = {"queries":len(correct),"correct":sum(correct)}
    for label, tag in (("direction","spatial"),("color","appearance")):
        groups = [g for g in pairs(ds).values() if tag in ds["raw"][g[0]]["query_types"]]
        output[label] = {"groups":len(groups),"four_correct":sum(all(correct[i] for i in g) for g in groups),
            "language_both":sum(all(correct[i] for i in g if ds["raw"][i]["language"]==lang)
                                for g in groups for lang in ("en","zh"))}
    return output


def calibrate(head, tune, base):
    components = ranking_components(head, tune["examples"])
    direction_grid = []
    for a,b in product((0.,2.,4.,8.,12.,20.), (0.,2.,4.)):
        weights = (a,b,0.)
        m = group_results(tune,add_components(base,components,weights))
        direction_grid.append({"weights":weights,"results":m})
    def direction_key(rec):
        r=rec["results"]["direction"]
        a,b,c=rec["weights"]
        return (r["language_both"],r["four_correct"],rec["results"]["correct"],-a,-b)
    direction = max(direction_grid,key=direction_key)
    a,b,_ = direction["weights"]
    color_grid = []
    for c in (0.,1.,2.,4.,8.):
        weights=(a,b,c)
        m=group_results(tune,add_components(base,components,weights))
        color_grid.append({"weights":weights,"results":m})
    def color_key(rec):
        r=rec["results"]["color"]
        return (r["language_both"],r["four_correct"],rec["results"]["correct"],-rec["weights"][2])
    color = max(color_grid,key=color_key)
    return {"direction":direction,"direction_color":color,"direction_grid":direction_grid,"color_grid":color_grid}


def scores_for(model, head, ds, lexicon, selection, weight):
    m2 = predict_scores(model, ds)
    b2, hybrid = [], []
    for ex, score in zip(ds["examples"],m2):
        if score is None:
            b2.append(None);hybrid.append(None)
        else:
            q = ex["query"]
            bonus = weight * position_scores(ex["candidates"],shared_parse(q.query,q.language,lexicon))
            b2.append(ex["base"]+bonus);hybrid.append(score+bonus)
    terms = ranking_components(head,ds["examples"])
    return {"M2-binding":("M2",m2),"B2-rule":("B2",b2),"M2-B2-rule":("M2",hybrid),
            "M2-learned-direction":("M2",add_components(m2,terms,selection["direction"]["weights"])),
            "M2-learned-attributes":("M2",add_components(m2,terms,selection["direction_color"]["weights"]))}


def run(a):
    start=time.monotonic()
    cfg=read_json(a.config)
    frozen=read_json(a.fp_config)
    require(cfg["datasets"]["fp_pred"]["candidates"] == frozen["candidates"],"stale FP candidate input")
    require(cfg["pins"][frozen["candidates"]]==frozen["sha256"]["candidates"],"FP candidate pin mismatch")
    check_release(frozen,{"candidates":Path(frozen["candidates"])},CandidatePack(frozen["candidates"]))
    verify(cfg)
    split_checks(cfg)
    reference=read_json(a.selection_reference/"summary.json")
    require(reference["status"]=="completed","incomplete prior control")
    original=read_json(a.selection_reference/"selection.json")
    chosen=next(x for x in original["history"] if x["update"]==original["chosen"]["fp_overall"])
    ckpt=Path(chosen["checkpoint"])
    require(ckpt.resolve().is_relative_to(a.selection_reference.resolve()) and sha256(ckpt)==chosen["checkpoint_sha256"],
            "changed base ranker checkpoint")
    prior_cfg=read_json(Path(cfg["previous_inference"])/"run_metadata.json")
    lexicon=read_json(cfg["lexicon"])
    legacy=read_json(Path(a.rule_selection))
    require(legacy["selected_weight"]==2 and legacy["lexicon_sha256"]==sha256(cfg["lexicon"]),"wrong frozen rule baseline")

    a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(SEED);np.random.seed(SEED);random.seed(SEED)
    builder=builder_for(cfg,a.output,[str(a.selection_reference)])
    builder.candidate_features=binding_geometry
    def no_encoding():
        raise RuntimeError("expected verified frozen features")
    builder.model=no_encoding

    data=build_inputs(builder,cfg,("train",))["train"]
    inc_ids={q["query_id"] for q in map(json.loads,Path(cfg["increment_train"]).read_text().splitlines())}
    train,tune=divide(data,inc_ids,a.output)
    split=read_json(a.output/"split.json")
    require(split==reference["split"] and builder.signature==prior_cfg["encoder"],"unexpected training split or encoder")
    trunk=model_at(ckpt,builder.signature)
    aux, fit_info=fit_heads(train,lexicon)
    selected=calibrate(aux,tune,predict_scores(trunk,tune))
    state_path=a.output/"attribute_head.pt"
    torch.save({"state":aux.state_dict(),"encoder":builder.signature,"seed":SEED,"base_checkpoint_sha256":sha256(ckpt),
                "selection":{k:selected[k]["weights"] for k in ("direction","direction_color")},
                "training_query_sha256":train["hashes"]["queries"]},state_path)
    reload=AttributeHeads(builder.signature["dimension"])
    reload.load_state_dict(torch.load(state_path,map_location="cpu")["state"],strict=True)
    before=ranking_components(aux,tune["examples"])
    after=ranking_components(reload.eval(),tune["examples"])
    require(all(np.array_equal(x,y) for p,q in zip(before,after) for x,y in zip(p,q)), "head replay differs")
    write_json(a.output/"calibration.json",selected)
    print(json.dumps({"selected": {k:selected[k] for k in ("direction","direction_color")},"fit":fit_info}),flush=True)
    del data,train,tune

    outputs = {}
    for name in ("fp_pred","mhp_gt","mhp_pred"):
        ds=build_inputs(builder,cfg,(name,))[name]
        path=a.output/name
        path.mkdir()
        results={}
        for label,(method,scores) in scores_for(trunk,aux,ds,lexicon,selected,legacy["selected_weight"]).items():
            result=publish_scores(ds,method,label,scores,path,{"run_id":a.output.name+"-"+name},draft=name.startswith("mhp"))
            subgroup=pair_metrics(ds,result["rows"]) if name.startswith("mhp") else None
            if name=="mhp_pred":
                require(subgroup["direction"]["both_targets_covered_pairs"]==130
                        and subgroup["color"]["both_targets_covered_pairs"]==42,"candidate coverage changed")
            results[label]={"summary":result["summary"],"pairs":subgroup}
        outputs[name]={"hashes":ds["hashes"],"methods":results}
        write_json(a.output/"partial_results.json",outputs)
        print(json.dumps({"done":name,"methods":results}),flush=True)

    write_json(a.output/"summary.json",{"status":"completed","git_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "command":[sys.executable,*sys.argv],"config_sha256":sha256(a.config),"fp_config_sha256":sha256(a.fp_config),
        "attributes_sha256":sha256(state_path),"base_checkpoint_sha256":sha256(ckpt),"split":split,"fit":fit_info,
        "calibration":selected,"conditions":outputs,"rule_weight":legacy["selected_weight"],
        "cache_misses":builder.encoded_images+builder.encoded_texts,"seconds":time.monotonic()-start})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("config","fp-config","rule-selection","selection-reference","output"):
        p.add_argument("--"+key,type=Path,required=True)
    run(p.parse_args())
