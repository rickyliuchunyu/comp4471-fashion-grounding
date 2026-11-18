"""Controlled Fashionpedia/MHP binding study with rule-free model selection."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import torch

from fashion_grounding.data.schema import read_json, require, sha256, write_json
from fashion_grounding.grounding.binding import binding_geometry
from fashion_grounding.grounding.controlled import fit, predict_scores, publish_scores, supervised_samples
from fashion_grounding.grounding.pair_objective import PairMargin
from fashion_grounding.grounding.spatial import make_head
from scripts.prepare_mhp_binding_features import build_inputs, builder_for, verify

SETTINGS = {"seed": 42, "updates": 490, "batch_size": 32, "evaluation_interval": 98}
METHODS = {"M2-mixed-CE": "M2", "M2-mixed-pair": "M2", "M2-binding-pair": "M2-binding"}


def split_checks(config):
    ds = config["datasets"]
    manifests = {
        "train": ds["train"]["manifest"], "fp_dev": ds["fp_gt"]["manifest"],
        "mhp_dev": ds["mhp_gt"]["manifest"], "test": config["protected_test_manifest"],
    }
    records = {k: read_json(p)["images"] for k, p in manifests.items()}
    for key in records:
        for other in records:
            if other >= key:
                continue
            for field in ("image_id", "sha256", "group_id"):
                require(not {x[field] for x in records[key]} & {x[field] for x in records[other]},
                        f"cross-split overlap: {key}/{other}/{field}")


def pairs(data, ids=None):
    grouped = defaultdict(list)
    for idx, q in enumerate(data["raw"]):
        if q.get("contrast_set_id") and (ids is None or q["query_id"] in ids):
            grouped[q["contrast_set_id"]].append(idx)
    require(bool(grouped), "no paired queries")
    for members in grouped.values():
        qs = [data["raw"][i] for i in members]
        require(len(qs) == 4 and len({q["image_id"] for q in qs}) == 1
                and len({q["target_category_id"] for q in qs}) == 1
                and len({tuple(q["query_types"]) for q in qs}) == 1, "invalid four-query group")
        require(Counter(q["language"] for q in qs) == {"en": 2, "zh": 2}, "missing paired translation")
        by_language = [{q["target_instance_id"] for q in qs if q["language"] == lang} for lang in ("en", "zh")]
        require(len(by_language[0]) == 2 and by_language[0] == by_language[1], "mismatched target pair")
    return dict(grouped)


def counter_map(data, groups):
    samples = supervised_samples(data)
    counters = {}
    for indices in groups.values():
        for lang in ("en", "zh"):
            i, j = [i for i in indices if data["raw"][i]["language"] == lang]
            require(samples[i]["candidates"] == samples[j]["candidates"], "mismatched candidate order")
            a, b = samples[i]["target"], samples[j]["target"]
            require(a != b, "pair collapsed to one target")
            counters[i], counters[j] = b, a
    return counters


def sequence(rows, seed):
    require(bool(rows), "empty sampling pool")
    gen = torch.Generator().manual_seed(seed)
    while True:
        for i in torch.randperm(len(rows), generator=gen).tolist():
            yield rows[i]


def make_order(data, groups, settings, only_pairs=False):
    used = {idx for group in groups.values() for idx in group}
    if only_pairs:
        stream = sequence(list(groups.values()), settings["seed"])
        return [i for _ in range(settings["updates"] * 8) for i in next(stream)]
    pos, col = [], []
    for members in groups.values():
        (pos if "spatial" in data["raw"][members[0]]["query_types"] else col).append(members)
    require(len(pos) == 320 and len(col) == 103, "wrong MHP incremental pair groups")
    position, color = sequence(pos, 43), sequence(col, 44)
    parent = sequence([i for i, e in enumerate(data["examples"]) if i not in used and len(e["visual"]) > 1], 45)
    order = []
    for _ in range(settings["updates"]):
        for _ in range(2):
            order.extend(next(position))
            order.extend(next(color))
        order.extend(next(parent) for _ in range(16))
    return order


def project_geometry(data, method):
    if method == "M2-binding":
        return data
    return {**data, "examples": [{**e, "geometry": e["geometry"][:, :12].copy()} for e in data["examples"]]}


def fit_model(method, data, dev, name, out, signature, settings, order=None, objective=None):
    model, info = fit(settings, method, project_geometry(data, method), project_geometry(dev, method),
                      out / name, signature, order=order, auxiliary_loss=objective)
    return model, info


def pair_metrics(data, eval_rows):
    rows = {r["query_id"]: r for r in eval_rows}
    grouped = pairs(data)
    output = {}
    for label, kind in (("direction", "spatial"), ("color", "appearance")):
        gs = [members for members in grouped.values() if kind in data["raw"][members[0]]["query_types"]]
        panel = [[rows[data["raw"][i]["query_id"]] for i in members] for members in gs]
        both_visible = [rs for rs in panel if all(q["covered"] for q in rs)]
        strict_all = lambda rs: all(q["correct"] for q in rs)
        output[label] = {
            "pairs": len(panel), "all_four_correct": sum(map(strict_all, panel)),
            "language_pair_correct": sum(all(q["correct"] for q in rs if q["language"] == lang)
                                         for rs in panel for lang in ("en", "zh")),
            "single_correct": sum(q["correct"] for rs in panel for q in rs),
            "mask_success": sum(q["success_at_0_5"] for rs in panel for q in rs),
            "both_targets_covered_pairs": len(both_visible),
            "all_four_when_covered": sum(map(strict_all, both_visible)),
        }
    return output


def score_and_publish(model, method, data, name, out, run_id, draft):
    view = project_geometry(data, method)
    values = predict_scores(model, view)
    result = publish_scores(view, "M2", name, values, out, {"run_id": run_id}, draft=draft)
    return {"summary": result["summary"], "pairs": pair_metrics(view, result["rows"]) if draft else None}


def fit_sanity(train, full_groups, out, signature):
    # Fixed choices from known annotated groups, independent of any model outcome.
    picked, seen_images = {}, set()
    for source, kind, limit in (("fashionpedia", "spatial", 4), ("mhp_v2", "spatial", 4), ("mhp_v2", "appearance", 4)):
        candidates = [pos for pos in full_groups.values()
                      if train["raw"][pos[0]]["source_dataset"] == source
                      and kind in train["raw"][pos[0]]["query_types"]]
        candidates.sort(key=lambda pos: train["raw"][pos[0]]["image_id"])
        count = 0
        for indices in candidates:
            img = train["raw"][indices[0]]["image_id"]
            if img in seen_images:
                continue
            seen_images.add(img)
            picked[img] = indices
            count += 1
            if count == limit:
                break
        require(count == limit, "insufficient disjoint fit samples")
    all_indices = [i for group in picked.values() for i in group]
    small = {**train, "raw": [train["raw"][i] for i in all_indices],
             "examples": [train["examples"][i] for i in all_indices],
             "queries": [train["queries"][i] for i in all_indices]}
    small["hashes"] = {**train["hashes"], "queries": hashlib.sha256(
        json.dumps(small["raw"], sort_keys=True).encode()).hexdigest()}
    write_json(out / "sanity_rows.json", [q["query_id"] for q in small["raw"]])
    groups = pairs(small)
    order = make_order(small, groups, SETTINGS, only_pairs=True)
    aux = PairMargin(counter_map(small, groups), "full", 1.0, 2.0)
    result = {}
    for method in ("M2", "M2-binding"):
        model, info = fit_model(method, small, small, "sanity-" + method, out, signature, SETTINGS, order, aux)
        scores = predict_scores(model, project_geometry(small, method))
        samples = supervised_samples(small)
        correct = [int(scores[i].argmax()) == e["target"] for i, e in enumerate(samples)]
        result[method] = {"training_queries": len(correct), "correct": sum(correct),
            "all_four_correct": sum(all(correct[i] for i in g) for g in groups.values()),
            "pairs": len(groups), "checkpoint": info["checkpoint"], "checkpoint_sha256": info["checkpoint_sha256"]}
    return result


def run(a):
    started = time.monotonic()
    config = read_json(a.config)
    verify(config)
    split_checks(config)
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    builder = builder_for(config, a.output, [str(a.train_features), str(a.dev_features)])
    builder.candidate_features = binding_geometry
    def cache_only():
        raise RuntimeError("verified frozen feature is missing")
    builder.model = cache_only

    inputs = build_inputs(builder, config, ("train", "fp_gt"))
    train, dev = inputs["train"], inputs["fp_gt"]
    increment_ids = {q["query_id"] for q in map(json.loads, Path(config["increment_train"]).read_text().splitlines())}
    groups = pairs(train, increment_ids)
    require({train["raw"][i]["query_id"] for g in groups.values() for i in g} == increment_ids, "incomplete MHP increment")
    sanity = fit_sanity(train, pairs(train), a.output, builder.signature)
    print(json.dumps({"small_fit_training_only":sanity}),flush=True)
    write_json(a.output / "sanity.json", sanity)

    prior = read_json(config["fp_checkpoint_meta"])
    require(prior["seed"] == SETTINGS["seed"] and prior["optimizer_updates"] == SETTINGS["updates"]
            and sha256(prior["checkpoint"]) == prior["checkpoint_sha256"], "changed reference checkpoint")
    state = torch.load(prior["checkpoint"], map_location="cpu")
    require(state["encoder"] == builder.signature, "reference encoder mismatch")
    reference = make_head("M2", state["dimension"])
    reference.load_state_dict(state["model"], strict=True)
    models = {"M2-FP": (reference, "M2", prior)}
    order = make_order(train, groups, SETTINGS)
    objective = PairMargin(counter_map(train, groups), "full", 1.0, 2.0)
    for name, method in METHODS.items():
        uses_pair = name.endswith("-pair")
        settings = {**SETTINGS, "loss": "CE+same-image pair margin" if uses_pair else "CE",
                    "margin": 1.0 if uses_pair else None, "margin_weight": 2.0 if uses_pair else None}
        model, info = fit_model(method, train, dev, name, a.output, builder.signature, settings,
                                order if uses_pair else None, objective if uses_pair else None)
        models[name] = (model, method, info)
    require(models["M2-mixed-pair"][2]["training_order_sha256"] == models["M2-binding-pair"][2]["training_order_sha256"],
            "paired methods have different sampling")

    results = {}
    fingerprints = {k: ds["hashes"] for k, ds in inputs.items()}
    del train, dev, inputs
    for name in ("fp_gt", "fp_pred", "mhp_gt", "mhp_pred"):
        data = build_inputs(builder, config, (name,))[name]
        fingerprints[name] = data["hashes"]
        folder = a.output / name
        folder.mkdir()
        results[name] = {}
        for label, (model, method, _) in models.items():
            results[name][label] = score_and_publish(model, method, data, label, folder,
                                                     a.output.name + "-" + name, draft=name.startswith("mhp"))
            if name.startswith("mhp"):
                g = results[name][label]["pairs"]
                require(g["direction"]["pairs"] == 200 and g["color"]["pairs"] == 59, "wrong study pair denominator")
                if name=="mhp_pred":
                    require(g["direction"]["both_targets_covered_pairs"] == 130
                            and g["color"]["both_targets_covered_pairs"] == 42, "published coverage differs")
        write_json(a.output / "partial_results.json", results)
        print(json.dumps({"completed_condition":name,"results":results[name]}),flush=True)
    write_json(a.output / "summary.json", {
        "status": "completed", "git_commit": subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "config_sha256": sha256(a.config), "command": [sys.executable,*sys.argv], "settings": SETTINGS,
        "models":{k: m[2] for k,m in models.items()}, "fingerprints":fingerprints,
        "sanity":sanity, "results":results, "cache_misses":builder.encoded_images + builder.encoded_texts,
        "seconds":time.monotonic() - started,
    })


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("config", "train-features", "dev-features", "output"):
        p.add_argument("--"+key,type=Path,required=True)
    run(p.parse_args())
