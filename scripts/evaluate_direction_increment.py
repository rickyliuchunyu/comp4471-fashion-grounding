"""Controlled comparison of the 1,235-image direction-pair M2 and its 1,149-image parent."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

from fashion_grounding.data.schema import read_json, require, sha256, write_json
from fashion_grounding.grounding.cache import FeatureCache
from fashion_grounding.grounding.controlled import FeatureBuilder, fit, manifest_separation, predict_scores, publish_scores
from fashion_grounding.grounding.rules import position_scores, shared_parse
from fashion_grounding.grounding.spatial import make_head
from scripts.migrate_identical_grounding_features import run as migrate


def load_parent(experiment, signature):
    info = read_json(experiment / "run_metadata.json")["models"]["M2-expanded1149"]
    require(info["seed"] == 42 and info["optimizer_updates"] == 490 and info["batch_size"] == 32, "parent budget differs")
    path = Path(info["checkpoint"])
    require(sha256(path) == info["checkpoint_sha256"], "invalid parent weights")
    state = torch.load(path, map_location="cpu")
    require(state["method"] == "M2" and state["encoder"] == signature, "incompatible parent")
    model = make_head("M2", state["dimension"])
    model.load_state_dict(state["model"], strict=True)
    return model, info


def scores_for(dataset, parent, trained, lexicon, weight):
    old, new = (predict_scores(model, dataset) for model in (parent, trained))
    b2, old_prior, new_prior = [], [], []
    for example, before, after in zip(dataset["examples"], old, new):
        query = example["query"]
        bonus = weight * position_scores(example["candidates"], shared_parse(query.query, query.language, lexicon))
        b2.append(example["base"] + bonus if before is not None else None)
        old_prior.append(before + bonus if before is not None else None)
        new_prior.append(after + bonus if after is not None else None)
    return {"B2": b2, "M2-1149": old, "M2-1235": new, "M2-1149-prior": old_prior, "M2-1235-prior": new_prior}


def draft_results(dataset, scores, directory, paired, *, scope="unreviewed-dev agreement on GT candidates"):
    # Produce selections for all methods before reading the draft target label.
    selections = {}
    for label, values in scores.items():
        selections[label] = [e["candidates"][int(np.argmax(s))].candidate_id if s is not None else None
                             for e, s in zip(dataset["examples"], values)]
    pack = dataset["pack"]
    rows = []
    for index, q in enumerate(dataset["raw"]):
        targets = [c["candidate_id"] for c in pack.rows[q["image_id"]]["candidates"]
                   if c["source_annotation_id"] == q["target_instance_id"]]
        require(len(targets) == 1, "expected one GT instance in draft diagnostic")
        picks = {label: ids[index] for label, ids in selections.items()}
        rows.append({"query_id": q["query_id"], "image_id": q["image_id"], "language": q["language"],
                     "contrast_set_id": q.get("contrast_set_id"), "draft_target": targets[0],
                     "shoe": q["target_category_id"] == 6, "query_types": q["query_types"],
                     "selected": picks, "agrees": {k: v == targets[0] for k, v in picks.items()}})
    groups = {"all": rows, "shoe": [q for q in rows if q["shoe"]], "non_shoe": [q for q in rows if not q["shoe"]]}
    for kind in ("appearance", "combined", "spatial"):
        groups[kind] = [q for q in rows if kind in q["query_types"]]
    summary = {}
    for name, group in groups.items():
        images = defaultdict(list)
        for row in group:
            images[row["image_id"]].append(row)
        if paired:
            require(all(len(rs) == 4 and len({q["draft_target"] for q in rs}) == 2
                        and len({q["contrast_set_id"] for q in rs}) == 1 and rs[0]["contrast_set_id"]
                        and Counter(q["language"] for q in rs) == {"en": 2, "zh": 2}
                        and all(len({q["draft_target"] for q in rs if q["language"] == lang}) == 2 for lang in ("zh", "en"))
                        for rs in images.values()), "invalid paired diagnostic")
        methods = {}
        for label in scores:
            stats = {"agreement": sum(q["agrees"][label] for q in group)}
            if paired:
                stats["all_four_correct"] = sum(all(q["agrees"][label] for q in rs) for rs in images.values())
                stats["both_targets_correct_by_language"] = {
                    lang: sum(all(q["agrees"][label] for q in rs if q["language"] == lang) for rs in images.values())
                    for lang in ("en", "zh")
                }
                stats["same_instance_by_language"] = {
                    lang: sum(len({q["selected"][label] for q in rs if q["language"] == lang}) == 1 for rs in images.values())
                    for lang in ("en", "zh")
                }
            methods[label] = stats
        summary[name] = {"queries": len(group), "images": len(images), "methods": methods}
    directory.mkdir()
    write_json(directory / "draft_diagnostic.json", {"scope": scope,
               "candidate_set_id": pack.package["candidate_set_id"], "hashes": dataset["hashes"], "rows": rows, "summary": summary})
    return summary


COLOR = re.compile(r"\b(?:black|white|red|blue|green|yellow|pink|purple|brown|orange|beige|tan|grey|gray|cream|navy|gold|silver|burgundy|teal|khaki|maroon|olive)\b", re.I)


def reviewed_groups(dataset, rows):
    # Define membership on English and include both translations. Color mentions may coexist with other attributes.
    english = {q["semantic_query_id"]: q for q in dataset["raw"] if q["language"] == "en"}
    require(all(q["semantic_query_id"] in english for q in rows), "missing English pair")
    def has_color(q):
        return bool(COLOR.search(english[q["semantic_query_id"]]["query"]))

    groups = {
        "color_appearance": [q for q in rows if has_color(q) and "appearance" in q["query_types"]],
        "color_and_direction": [q for q in rows if has_color(q) and "combined" in q["query_types"]],
        "pure_direction": [q for q in rows if "spatial" in q["query_types"]],
    }
    groups["color_appearance_same_category"] = [q for q in groups["color_appearance"] if q["same_category_instance_count"] > 1]
    return {name: {"queries": len(group), "images": len({q["image_id"] for q in group}),
                   "correct": sum(q["correct"] for q in group), "mask_success": sum(q["success_at_0_5"] for q in group)}
            for name, group in groups.items()}


def run(config_file, output):
    start = time.monotonic()
    config = read_json(config_file)
    for path, digest in config["pins"].items():
        require(sha256(path) == digest, "changed input: " + path)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    previous = Path(config["previous_experiment"])
    builder = FeatureBuilder(config, output)
    require((previous / "features").is_dir(), "missing reference feature cache")
    builder.readonly = FeatureCache(previous / "features")
    paths = {k: Path(v) for k, v in config["packages"].items()}
    checks = manifest_separation(config["packages"])
    checks += manifest_separation({**config["packages"], "challenge": str(paths["direction74"])})
    for key in ("image_id", "sha256", "group_id"):
        def identities(name):
            return {row[key] for row in read_json(paths[name] / "image_manifest.json")["images"]}
        require(not identities("direction74") & (identities("dev") | identities("challenge") | identities("test")),
                "direction74 overlaps existing dev, challenge or test")
    write_json(output / "split_checks.json", checks)
    parent_queries = paths["clean1000"] / "queries_machine_train.jsonl"
    train_queries = paths["expanded1149"] / "queries_machine_train.jsonl"
    require(train_queries.read_bytes() == parent_queries.read_bytes() + (paths["expanded1149"] / "queries_increment.jsonl").read_bytes(),
            "training queries are not parent plus the frozen increment")
    migrated = migrate(SimpleNamespace(
        source=paths["clean1000"] / "candidates_gt.json", destination=paths["expanded1149"] / "candidates_gt.json",
        encoder_metadata=Path(config["previous_inference"]) / "run_metadata.json", cache=output / "features",
        source_cache=previous / "features", output=output / "cache_migration.json"))

    def build(name, query_file, training=False, draft=False, candidate_path=None):
        root = paths[name]
        return builder.build(name, candidate_path or root / "candidates_gt.json", root / query_file,
                             root / "annotations_coco.json", training=training, draft_dev=draft)

    train = build("expanded1149", "queries_machine_train.jsonl", training=True)
    dev = build("dev", "queries_accepted.jsonl")
    require(len(train["queries"]) == 3640 and len(dev["queries"]) == 642, "unexpected experiment size")
    parent, parent_info = load_parent(previous, builder.signature)
    require(parent_info["train_inputs"]["queries"] == sha256(parent_queries), "parent checkpoint uses different supervision")
    settings = {"seed": 42, "updates": 490, "batch_size": 32, "evaluation_interval": 98}
    model, model_info = fit(settings, "M2", train, dev, output / "M2-1235", builder.signature)
    lexicon = read_json(config["lexicon"])
    weight = read_json(previous / "B2-dev-selection.json")["selected_weight"]
    require(weight == 2, "frozen B2 prior changed")
    summary = {}
    predicted = build("dev", "queries_accepted.jsonl", candidate_path=Path(config["predicted_candidates"]))
    for condition, dataset in (("gt", dev), ("predicted", predicted)):
        summary[condition] = {}
        for label, scores in scores_for(dataset, parent, model, lexicon, weight).items():
            method = "M2+B2" if label.endswith("prior") else label.split("-")[0]
            result = publish_scores(dataset, method, label, scores, output, config)
            summary[condition][label] = {"summary": result["summary"], "groups": reviewed_groups(dataset, result["rows"])}
            if label in ("M2-1149", "B2"):
                prior_output = previous if condition == "gt" else Path(config["previous_50k"])
                prior_label = "M2-expanded1149" if label == "M2-1149" else "B2-selected"
                prior = read_json(prior_output / condition / f"{prior_label}-evaluation.json")
                require(result["summary"] == prior["summary"], "frozen reviewed comparison diverged: " + label)
    for name, expected in (("direction74", 296), ("challenge", 78)):
        draft = build(name, "queries_draft.jsonl", draft=True)
        require(len(draft["queries"]) == expected, "wrong draft denominator")
        summary[name] = draft_results(draft, scores_for(draft, parent, model, lexicon, weight),
                                      output / name, paired=name == "direction74")

    write_json(output / "summary.json", summary)
    write_json(output / "run_metadata.json", {
        "status": "completed", "git_commit": subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "command": [sys.executable, *sys.argv], "config_sha256": sha256(config_file),
        "pins": config["pins"], "encoder": builder.signature, "rule_weight": weight,
        "new_model": model_info, "parent_model": parent_info, "migrated_images": migrated,
        "encoded_images": builder.encoded_images, "encoded_texts": builder.encoded_texts,
        "seconds": time.monotonic() - start,
    })
    print(json.dumps({"status": "completed", "seconds": time.monotonic() - start}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.output)
