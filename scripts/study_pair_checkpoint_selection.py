"""Isolate checkpoint-selection criteria on a group-disjoint MHP tuning split."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import torch

from fashion_grounding.data.schema import read_json, require, sha256, write_json
from fashion_grounding.grounding.binding import binding_geometry
from fashion_grounding.grounding.controlled import predict_scores, supervised_samples
from fashion_grounding.grounding.pair_objective import PairMargin
from fashion_grounding.grounding.spatial import make_head
from scripts.prepare_mhp_binding_features import build_inputs, builder_for, verify
from scripts.run_mhp_binding_study import (
    SETTINGS, counter_map, fit_model, make_order, pairs, score_and_publish, split_checks,
)


def subset(data, keep, query_path):
    indices = [i for i, q in enumerate(data["raw"]) if keep(q)]
    filtered = {key: [data[key][i] for i in indices] for key in ("raw", "examples", "queries")}
    query_path.write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in filtered["raw"]),
                          encoding="utf-8")
    return {**data, **filtered, "query_path": str(query_path),
            "hashes": {**data["hashes"], "queries": sha256(query_path)}}


def divide(train, increment_ids, output):
    groups = pairs(train, increment_ids)
    manifest = train["pack"].images
    direction, color = set(), set()
    for ix in groups.values():
        q = train["raw"][ix[0]]
        require(q["source_dataset"] == "mhp_v2", "expected MHP increment")
        group = manifest[q["image_id"]]["group_id"]
        (direction if "spatial" in q["query_types"] else color).add(group)
    require(len(direction) == 320 and len(color) == 103 and color <= direction, "unexpected increment pair structure")

    def choose(pool):
        ordered = sorted(pool, key=lambda g: (hashlib.sha256(f"mhp-tuning-42:{g}".encode()).hexdigest(), str(g)))
        return set(ordered[:32])

    held = choose(color) | choose(direction - color)
    membership = lambda q: manifest[q["image_id"]]["group_id"] in held
    fit_data = subset(train, lambda q: not membership(q), output / "queries_fit.jsonl")
    tune_data = subset(train, membership, output / "queries_tuning.jsonl")
    ids = lambda ds: {q["image_id"] for q in ds["raw"]}
    require(len(held) == 64 and len(ids(tune_data)) == 64, "expected 64 isolated images")
    for field in ("image_id", "sha256", "group_id"):
        left = {manifest[i][field] for i in ids(fit_data)}
        right = {manifest[i][field] for i in ids(tune_data)}
        require(not left & right, "fit/tuning overlap: " + field)
    require(len(fit_data["raw"]) == 5220 and len(tune_data["raw"]) == 384, "unexpected tuning row counts")
    write_json(output / "split.json", {
        "selected_by": "deterministic SHA-256 order within color-present and direction-only strata; 32 groups each",
        "split_seed": 42, "held_groups": sorted(held), "held_images": sorted(ids(tune_data)),
        "fit_images": len(ids(fit_data)), "fit_queries": len(fit_data["raw"]), "fit_sha256": fit_data["hashes"]["queries"],
        "tuning_images": len(ids(tune_data)), "tuning_queries": len(tune_data["raw"]),
        "tuning_sha256": tune_data["hashes"]["queries"],
    })
    return fit_data, tune_data


def tuning_metrics(model, data):
    scores = predict_scores(model, data)
    samples = supervised_samples(data)
    correct = [value is not None and int(value.argmax()) == item["target"]
               for value, item in zip(scores, samples)]
    result = {"single_correct": sum(correct), "queries": len(correct)}
    groups = pairs(data)
    for label, kind in (("direction", "spatial"), ("color", "appearance")):
        panel = [g for g in groups.values() if kind in data["raw"][g[0]]["query_types"]]
        result[label] = {
            "groups": len(panel),
            "single_correct": sum(correct[i] for g in panel for i in g),
            "both_same_language": sum(all(correct[i] for i in g if data["raw"][i]["language"] == lang)
                                      for g in panel for lang in ("en", "zh")),
            "all_four_correct": sum(all(correct[i] for i in g) for g in panel),
        }
    require(result["direction"]["groups"] == 64 and result["color"]["groups"] == 32, "wrong tuning denominators")
    return result


def choose_updates(history):
    # Compare the two metrics on the same tuning set; earliest update wins exact ties.
    def pick(key):
        return max(history, key=lambda row: (key(row), -row["update"]))["update"]

    return {
        "fp_overall": pick(lambda row: row["fp_correct"]),
        "mhp_overall": pick(lambda row: row["tuning"]["single_correct"]),
        "mhp_direction_pair": pick(lambda row: row["tuning"]["direction"]["both_same_language"]),
    }


def model_at(path, signature):
    state = torch.load(path, map_location="cpu")
    require(state["method"] == "M2-binding" and state["encoder"] == signature, "checkpoint scoring contract mismatch")
    model = make_head(state["method"], state["dimension"])
    model.load_state_dict(state["model"], strict=True)
    return model


def run(args):
    began = time.monotonic()
    cfg = read_json(args.config)
    verify(cfg)
    split_checks(cfg)
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    builder = builder_for(cfg, args.output, [str(path) for path in args.reference])
    builder.candidate_features = binding_geometry
    def unexpected_encoding():
        raise RuntimeError("expected verified cache for every feature")
    builder.model = unexpected_encoding

    data = build_inputs(builder, cfg, ("train", "fp_gt"))
    increment_ids = {q["query_id"] for q in map(json.loads, Path(cfg["increment_train"]).read_text().splitlines())}
    train, tuning = divide(data["train"], increment_ids, args.output)
    training_pairs = pairs(train, increment_ids)
    order = make_order(train, training_pairs, SETTINGS, expected_pair_counts=(256, 71))
    aux = PairMargin(counter_map(train, training_pairs), "full", 1.0, 2.0)
    settings = {**SETTINGS, "loss":"CE+same-image pair margin", "margin":1.0, "margin_weight":2.0}
    ckpts = args.output / "M2-binding-fit"
    _, meta = fit_model("M2-binding", train, data["fp_gt"], ckpts.name,
                        args.output, builder.signature, settings, order, aux)

    hist = []
    for row in read_json(ckpts / "training_log.json"):
        update = row["update"]
        p = ckpts / f"update_{update:04d}.pt"
        hist.append({"update": update, "fp_correct": row["dev_correct"],
                     "tuning": tuning_metrics(model_at(p,builder.signature), tuning),
                     "checkpoint": str(p), "checkpoint_sha256": sha256(p)})
    selected = choose_updates(hist)
    require(selected["fp_overall"] == meta["selected_update"], "reviewed-dev selection mismatch")
    write_json(args.output / "selection.json", {"chosen": selected, "history": hist,
               "scope": "selections fixed before scoring original MHP diagnostic development inputs"})
    print(json.dumps({"selection": selected, "history": hist}), flush=True)
    live_models = {u: model_at(ckpts / f"update_{u:04d}.pt",builder.signature) for u in sorted(set(selected.values()))}

    reviewed_gt = data["fp_gt"]["hashes"]
    del train, tuning, data, order
    evaluation = {}
    for name in ("fp_pred", "mhp_gt", "mhp_pred"):
        ds = build_inputs(builder, cfg, (name,))[name]
        folder = args.output / name
        folder.mkdir()
        results = {}
        for update, model in live_models.items():
            results[str(update)] = score_and_publish(model, "M2-binding", ds, f"update-{update:04d}",
                folder, args.output.name + "-" + name, draft=name.startswith("mhp"))
            if name == "mhp_pred":
                result = results[str(update)]["pairs"]
                require(result["direction"]["both_targets_covered_pairs"] == 130
                        and result["color"]["both_targets_covered_pairs"] == 42, "MHP candidate coverage changed")
        evaluation[name] = {"hashes": ds["hashes"], "by_update": results}
        write_json(args.output / "partial_results.json", evaluation)
        print(json.dumps({"completed": name, "results": results}), flush=True)

    write_json(args.output / "summary.json", {
        "status":"completed", "git_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "config_sha256":sha256(args.config), "command":[sys.executable,*sys.argv],
        "fit":meta, "split":read_json(args.output/"split.json"), "selection":selected, "history":hist,
        "reviewed_gt_hashes":reviewed_gt, "evaluation":evaluation,
        "encoded_regions":builder.encoded_images, "encoded_texts":builder.encoded_texts,
        "seconds":time.monotonic()-began,
    })


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("config", "output"):
        p.add_argument("--" + arg, type=Path, required=True)
    p.add_argument("--reference", type=Path, action="append", required=True)
    run(p.parse_args())
