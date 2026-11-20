"""Score with a pinned classifier and base ranker without retraining."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import torch

from fashion_grounding.data.schema import read_json, require, sha256, write_json
from fashion_grounding.grounding.attributes import AttributeHeads, hybrid_scores
from fashion_grounding.grounding.binding import binding_geometry
from fashion_grounding.grounding.controlled import predict_scores, publish_scores
from fashion_grounding.grounding.inputs import CandidatePack
from scripts.evaluate_frozen_grounding_dev import check_release
from scripts.prepare_mhp_binding_features import build_inputs, builder_for, verify
from scripts.run_mhp_binding_study import pair_metrics
from scripts.study_pair_checkpoint_selection import model_at

CONDITIONS = ("fp_pred", "mhp_gt", "mhp_pred")


def pinned(spec):
    path = Path(spec["path"])
    require(sha256(path) == spec["sha256"], "release file changed: " + str(path))
    return path


def run(a):
    start = time.monotonic()
    spec = read_json(a.release)
    cfg = read_json(pinned(spec["resolved_config"]))
    frozen = read_json(pinned(spec["frozen_fp_config"]))
    require(a.candidates == Path(frozen["candidates"]) == Path(cfg["datasets"]["fp_pred"]["candidates"]),
            "--candidates does not match the frozen Fashionpedia release")
    check_release(frozen, {"candidates":a.candidates}, CandidatePack(a.candidates))
    verify(cfg)
    base_path = pinned(spec["base_checkpoint"])
    attribute_path = pinned(spec["attribute_head"])
    state = torch.load(attribute_path, map_location="cpu")
    require(state["base_checkpoint_sha256"]==spec["base_checkpoint"]["sha256"]
            and state["selection"]["direction_color"] == spec["weights"], "attribute/base release mismatch")
    rule = read_json(pinned(spec["rule_selection"]))
    require(rule["selected_weight"] == 2 and rule["lexicon_sha256"] == sha256(cfg["lexicon"]), "rule selection changed")
    previous = read_json(pinned(spec["study_summary"]))
    require(previous["status"]=="completed" and previous["attributes_sha256"]==spec["attribute_head"]["sha256"],
            "study provenance mismatch")

    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    builder = builder_for(cfg, a.output, [str(Path(spec["study_summary"]["path"]).parent)])
    builder.candidate_features = binding_geometry
    require(state["encoder"]==builder.signature, "wrong frozen encoder")
    model = model_at(base_path,builder.signature)
    head = AttributeHeads(builder.signature["dimension"])
    head.load_state_dict(state["state"],strict=True)
    head.eval()
    lexicon = read_json(cfg["lexicon"])
    outputs={}
    for name in a.conditions or CONDITIONS:
        ds = build_inputs(builder,cfg,(name,))[name]
        require(ds["hashes"] == previous["conditions"][name]["hashes"], "candidate/query identity changed")
        values = predict_scores(model,ds)
        scores, counts = hybrid_scores(head,ds["examples"],values,spec["weights"],lexicon,rule["selected_weight"])
        folder = a.output/name
        folder.mkdir()
        draft = name.startswith("mhp")
        result = publish_scores(ds,"M2","M2-hybrid-attributes",scores,folder,a.output.name+"-"+name,draft=draft)
        pair_summary = pair_metrics(ds,result["rows"]) if draft else None
        if name=="mhp_pred":
            require(pair_summary["direction"]["both_targets_covered_pairs"]==130
                    and pair_summary["color"]["both_targets_covered_pairs"]==42, "MHP candidate coverage changed")
        outputs[name]={"hashes":ds["hashes"],"routing":counts,"summary":result["summary"],"pairs":pair_summary}
        print(json.dumps({"done":name,**outputs[name]}),flush=True)
    write_json(a.output/"summary.json",{
        "status":"completed","git_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "command":[sys.executable,*sys.argv],"release_sha256":sha256(a.release),"outputs":outputs,
        "usage":"exploratory development composition chosen after comparing the constituent methods",
        "cache_misses":builder.encoded_images+builder.encoded_texts,"seconds":time.monotonic()-start})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("release","candidates","output"):
        p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--conditions",nargs="+",choices=CONDITIONS)
    run(p.parse_args())
