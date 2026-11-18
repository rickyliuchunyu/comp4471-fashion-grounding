"""Test data exposure and the geometry branch with all position rules disabled."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import subprocess
import sys

import torch

from fashion_grounding.data.schema import read_json, require, sha256, write_json
from fashion_grounding.grounding.cache import FeatureCache
from fashion_grounding.grounding.controlled import (
    FeatureBuilder, fit, gt_selection, predict_scores, training_order,
)
from scripts.evaluate_direction_increment import draft_results


def paired_order(raw, samples, increment_ids, settings):
    groups = defaultdict(list)
    parent = []
    for index, (query, sample) in enumerate(zip(raw, samples)):
        if len(sample["visual"]) < 2:
            require(query["query_id"] not in increment_ids, "increment target lacks competing candidates")
            continue
        if query["query_id"] in increment_ids:
            group = query["contrast_set_id"]
            require(bool(group), "paired query has no contrast group")
            groups[group].append(index)
        else:
            parent.append(index)
    pairs = list(groups.values())
    require(len(pairs) == 86 and all(len(pair) == 4 for pair in pairs), "expected 86 complete four-query groups")
    for pair in pairs:
        require(len({raw[i]["image_id"] for i in pair}) == 1, "pair spans multiple images")
        require(Counter(raw[i]["language"] for i in pair) == {"en": 2, "zh": 2}, "pair must be bilingual")
        require(all(len({samples[i]["target"] for i in pair if raw[i]["language"] == lang}) == 2
                    for lang in ("en", "zh")), "expected distinct targets by language")

    def shuffled_stream(items, offset):
        generator = torch.Generator().manual_seed(settings["seed"] + offset)
        while True:
            for i in torch.randperm(len(items), generator=generator).tolist():
                yield items[i]

    pair_stream = shuffled_stream(pairs, 1)
    parent_stream = shuffled_stream(parent, 2)
    order = []
    for _ in range(settings["updates"]):
        order.extend(i for _ in range(4) for i in next(pair_stream))
        order.extend(next(parent_stream) for _ in range(16))
    return order


def run(config_file, out, previous_run, device):
    config = read_json(config_file)
    for path, digest in config["pins"].items():
        require(sha256(path) == digest, "changed input: " + path)
    previous = read_json(previous_run / "run_metadata.json")
    require(previous["status"] == "completed", "incomplete reference study")
    require((previous_run/"features").is_dir(), "missing prior cache")
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    builder = FeatureBuilder(config, out)
    builder.readonly = FeatureCache(previous_run / "features")
    # The prior experiment built and validated these inputs. A cache miss must fail rather than downloading or encoding.
    def cache_only():
        raise RuntimeError("missing frozen feature cache")
    builder.model = cache_only

    paths = {key: Path(value) for key, value in config["packages"].items()}
    def build(key, filename, *, training=False, draft_dev=False, candidate_path=None):
        root = paths[key]
        return builder.build(key, candidate_path or root / "candidates_gt.json",
                             root / filename, root / "annotations_coco.json",
                             training=training, draft_dev=draft_dev)

    train = build("expanded1149", "queries_machine_train.jsonl", training=True)
    dev = build("dev", "queries_accepted.jsonl")
    require(train["hashes"] == previous["new_model"]["train_inputs"], "training dataset changed")
    require(dev["hashes"] == previous["new_model"]["dev_inputs"], "checkpoint-selection dataset changed")
    settings = {"seed": 42, "updates": 490, "batch_size": 32, "evaluation_interval": 98}
    from fashion_grounding.grounding.controlled import supervised_samples
    samples = supervised_samples(train)
    balanced_ids = {q["query_id"] for q in map(json.loads, (paths["expanded1149"] / "queries_increment.jsonl").read_text().splitlines())}
    uniform = training_order(samples, settings["seed"], settings["updates"], settings["batch_size"])
    paired = paired_order(train["raw"], samples, balanced_ids, settings)
    orders = {"M1-uniform": uniform, "M2-uniform": uniform, "M2-pairs50": paired}

    models, infos, exposures = {}, {}, {}
    for label, order in orders.items():
        method = label.split("-")[0]
        model, info = fit(settings, method, train, dev, out / label, builder.signature,
                          device=device, order=order)
        models[label] = model
        infos[label] = info
        exposures[label] = dict(
            total=len(order),
            direction_pair_queries=sum(train["raw"][i]["query_id"] in balanced_ids for i in order),
            pure_spatial_queries=sum("spatial" in train["raw"][i]["query_types"] for i in order),
        )
    require(infos["M1-uniform"]["training_order_sha256"] == infos["M2-uniform"]["training_order_sha256"],
            "M1/M2 samples differ")
    if device == "cuda":
        peak = torch.cuda.max_memory_allocated()
    else:
        peak = None
    draft = build("direction74", "queries_draft.jsonl", draft_dev=True)
    require(len(draft["queries"]) == 296, "unexpected draft scope")
    # All three checkpoints were selected on identical reviewed GT dev before opening the draft set.
    score_sets = {label: predict_scores(model, draft) for label, model in models.items()}
    summary = draft_results(draft, score_sets, out / "direction74", paired=True)
    dev_gt = {label: gt_selection(dev, predict_scores(model, dev)) for label, model in models.items()}
    write_json(out / "summary.json", {
        "git_commit": subprocess.check_output(["git","rev-parse","HEAD"], text=True).strip(),
        "command": [sys.executable, *sys.argv], "config_sha256": sha256(config_file),
        "device": device, "gpu_peak_allocated_bytes": peak, "settings": settings,
        "device_name": torch.cuda.get_device_name() if device=="cuda" else None,
        "model_checkpoints": infos, "exposures": exposures, "draft": summary,
        "reviewed_gt_selection_correct": dev_gt, "reviewed_queries": len(dev["queries"]),
        "cache": {"encoded_images": builder.encoded_images, "encoded_texts": builder.encoded_texts},
    })
    print(json.dumps({"main_dev":dev_gt,"draft_all":summary["all"],"draft_non_shoe":summary["non_shoe"],
                      "exposures":exposures,"peak_bytes":peak}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("config", "output", "previous-run"):
        parser.add_argument("--" + arg, type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    args = parser.parse_args()
    run(args.config, args.output, args.previous_run, args.device)
