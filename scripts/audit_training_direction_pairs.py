"""Measure rule-free fit on the 86 machine-labeled training contrast pairs."""
import argparse
import json
from pathlib import Path

import torch

from fashion_grounding.data.schema import read_json, require, sha256, write_json
from fashion_grounding.grounding.cache import FeatureCache
from fashion_grounding.grounding.controlled import FeatureBuilder, predict_scores
from fashion_grounding.grounding.spatial import make_head
from scripts.evaluate_direction_increment import draft_results


def run(config_path, completed, out):
    config = read_json(config_path)
    summary = read_json(completed / "summary.json")
    require(summary["config_sha256"] == sha256(config_path), "experiment config changed")
    root = Path(config["packages"]["expanded1149"])
    increment = root / "queries_increment.jsonl"
    require(sha256(increment) == config["pins"][str(increment)], "increment changed")
    require((completed / "features").is_dir(), "missing prior cache")
    torch.set_num_threads(4)
    out.mkdir(parents=True, exist_ok=False)

    # FeatureBuilder requires the explicit weak-label filename for all train-only inputs.
    subset = out / "queries_machine_train.jsonl"
    subset.write_bytes(increment.read_bytes())
    builder = FeatureBuilder(config, out)
    builder.readonly = FeatureCache(completed / "features")
    def cache_only():
        raise RuntimeError("missing frozen feature cache")
    builder.model = cache_only
    data = builder.build("train-increment", root / "candidates_gt.json", subset, root / "annotations_coco.json", training=True)
    require(len(data["queries"]) == 344, "unexpected training increment size")
    scores = {}
    for label, info in summary["model_checkpoints"].items():
        path = Path(info["checkpoint"])
        require(path.resolve().is_relative_to(completed.resolve()) and sha256(path)==info["checkpoint_sha256"],
                "checkpoint differs")
        state = torch.load(path, map_location="cpu")
        require(state["encoder"] == builder.signature, "encoder signature differs")
        model = make_head(info["method"], state["dimension"])
        model.load_state_dict(state["model"], strict=True)
        scores[label] = predict_scores(model, data)
    stats = draft_results(data, scores, out / "train_increment", paired=True,
                          scope="in-sample agreement with training labels on GT candidates")
    write_json(out / "summary.json", {"inputs": data["hashes"], "models": {k: r["checkpoint_sha256"]
               for k, r in summary["model_checkpoints"].items()}, "in_sample": stats})
    print(json.dumps(stats["all"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("config", "completed", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.completed, args.output)
