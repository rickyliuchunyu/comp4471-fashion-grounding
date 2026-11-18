"""Contrast two loss variants against the same-sampling M2 classification baseline."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch

from fashion_grounding.data.schema import read_json, require, sha256, write_json
from fashion_grounding.grounding.cache import FeatureCache
from fashion_grounding.grounding.controlled import FeatureBuilder, fit, predict_scores, supervised_samples
from fashion_grounding.grounding.pair_objective import PairMargin, counter_targets
from fashion_grounding.grounding.spatial import make_head
from scripts.diagnose_spatial_supervision import paired_order
from scripts.evaluate_direction_increment import draft_results


def build_cached(config, out, cache_run):
    builder = FeatureBuilder(config, out)
    require((cache_run / "features").is_dir(), "missing reference features")
    builder.readonly = FeatureCache(cache_run / "features")
    def cache_only():
        raise RuntimeError("missing previously verified frozen features")
    builder.model = cache_only
    return builder


def run(a):
    config = read_json(a.config)
    for path, digest in config["pins"].items():
        require(sha256(path) == digest, "changed input: " + path)
    prior = read_json(a.previous / "summary.json")
    require(prior["config_sha256"] == sha256(a.config), "reference used different data")
    require(prior["settings"] == {"seed": 42, "updates": 490, "batch_size": 32, "evaluation_interval": 98},
            "reference budget differs")
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    builder = build_cached(config, a.output, a.previous)
    roots = {key: Path(value) for key, value in config["packages"].items()}

    def dataset(key, query_file, *, training=False, draft=False):
        r = roots[key]
        return builder.build(key, r / "candidates_gt.json", query_file, r / "annotations_coco.json",
                             training=training, draft_dev=draft)

    train = dataset("expanded1149", roots["expanded1149"] / "queries_machine_train.jsonl", training=True)
    dev = dataset("dev", roots["dev"] / "queries_accepted.jsonl")
    baseline = prior["model_checkpoints"]["M2-pairs50"]
    require(train["hashes"] == baseline["train_inputs"] and dev["hashes"] == baseline["dev_inputs"], "matched inputs required")
    increment = roots["expanded1149"] / "queries_increment.jsonl"
    pair_ids = {q["query_id"] for q in map(json.loads, increment.read_text().splitlines())}
    samples = supervised_samples(train)
    counters = counter_targets(train["raw"], samples, pair_ids)
    settings = prior["settings"]
    order = paired_order(train["raw"], samples, pair_ids, settings)

    basestate = torch.load(baseline["checkpoint"], map_location="cpu")
    require(sha256(baseline["checkpoint"]) == baseline["checkpoint_sha256"]
            and basestate["encoder"] == builder.signature
            and basestate["training_order"] == order, "baseline identity mismatch")
    baseline_head = make_head("M2", basestate["dimension"])
    baseline_head.load_state_dict(basestate["model"], strict=True)
    models = {"M2-CE": baseline_head}
    reports = {"M2-CE": baseline}
    for score_source in ("full", "geometry"):
        label = "M2-pair-" + score_source
        objective = PairMargin(counters, score_source, margin=1.0, weight=2.0)
        model, info = fit({**settings, "pair_objective": {"score_source": score_source, "margin": 1.0, "weight": 2.0}},
                          "M2", train, dev, a.output / label, builder.signature,
                          device=a.device, order=order, auxiliary_loss=objective)
        require(info["training_order_sha256"] == baseline["training_order_sha256"], "training exposure differs")
        models[label], reports[label] = model, info

    draft = dataset("direction74", roots["direction74"] / "queries_draft.jsonl", draft=True)
    held = draft_results(draft, {k: predict_scores(model, draft) for k, model in models.items()},
                         a.output / "direction74", paired=True)
    require(held["all"]["methods"]["M2-CE"] == prior["draft"]["all"]["methods"]["M2-pairs50"],
            "frozen baseline draft changed")
    # Reuse already-validated training features and score only increment records.
    rows = [i for i, q in enumerate(train["raw"]) if q["query_id"] in pair_ids]
    subset = {**train, "raw": [train["raw"][i] for i in rows], "examples": [train["examples"][i] for i in rows],
              "hashes": {**train["hashes"], "queries": sha256(increment)}}
    insample = draft_results(subset, {k: predict_scores(model, subset) for k, model in models.items()},
                             a.output / "train86", paired=True,
                             scope="in-sample agreement with training labels on GT candidates")
    write_json(a.output / "summary.json", {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "command": [sys.executable, *sys.argv], "config_sha256": sha256(a.config),
        "models": reports, "draft": held, "train_increment": insample, "reference": str(a.previous),
        "gpu": torch.cuda.get_device_name() if a.device == "cuda" else None,
        "peak_bytes": torch.cuda.max_memory_allocated() if a.device == "cuda" else None,
        "runtime_versions": {"torch": torch.__version__, "cuda": torch.version.cuda},
    })
    print(json.dumps({"draft": held["all"], "draft_non_shoe": held["non_shoe"], "train": insample["all"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("config", "previous", "output"):
        parser.add_argument("--" + key, required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    run(parser.parse_args())
