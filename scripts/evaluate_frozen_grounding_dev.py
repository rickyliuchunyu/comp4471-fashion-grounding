"""Compare frozen grounding methods on a new, pinned development candidate pack."""
import argparse
import subprocess
import sys
from pathlib import Path

import torch

from fashion_grounding.data.schema import read_json, require, sha256, write_json
from fashion_grounding.grounding.controlled import FeatureBuilder, predict_scores, publish_scores
from fashion_grounding.grounding.rules import position_scores, rule_coverage, shared_parse
from fashion_grounding.grounding.spatial import make_head
from scripts.evaluate_predictions import evaluate_files
from scripts.run_b0 import run as run_b0


def run(config_path, output, run_id):
    config = read_json(config_path)
    paths = {key: Path(config[key]) for key in ("candidates", "queries", "annotations", "lexicon")}
    hashes = {key: sha256(path) for key, path in paths.items()}
    require(hashes == config["sha256"], "candidate, query, annotation or lexicon bytes changed")
    previous = Path(config["previous_experiment"])
    previous_meta = read_json(previous / "run_metadata.json")
    selection = read_json(previous / "B2-dev-selection.json")
    require(selection["query_sha256"] == hashes["queries"], "B2 selection used a different query snapshot")
    require(selection["lexicon_sha256"] == hashes["lexicon"], "B2 lexicon changed since selection")
    weight = selection["selected_weight"]

    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    builder = FeatureBuilder(config, output)
    require((previous / "features").is_dir(), "missing reference feature cache")
    from fashion_grounding.grounding.cache import FeatureCache
    builder.readonly = FeatureCache(previous / "features")
    metadata = {
        "scope": "reviewed development set with frozen heads",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "command": [sys.executable, *sys.argv],
        "config_sha256": sha256(config_path),
        "input_paths": {k: str(v) for k, v in paths.items()},
        "input_sha256": hashes,
        "encoder": builder.signature,
        "rule_weight": weight,
        "run_id": run_id,
        "status": "running",
    }
    write_json(output / "run_metadata.json", metadata)
    ds = builder.build("u50000-dev", paths["candidates"], paths["queries"], paths["annotations"])
    pack = ds["pack"]
    require(pack.package["candidate_source"] == "predicted", "predicted candidates required")
    require(pack.package["candidate_set_id"] == config["candidate_set_id"], "candidate set ID mismatch")
    require(len(pack.rows) == 250 and len(ds["queries"]) == 642, "unexpected scope")
    lexicon = read_json(paths["lexicon"])
    base = [e["base"] if e["candidates"] else None for e in ds["examples"]]
    scores = {
        "B1": base,
        "B2-selected": [
            e["base"] + weight * position_scores(
                e["candidates"], shared_parse(e["query"].query, e["query"].language, lexicon)
            ) if e["candidates"] else None
            for e in ds["examples"]
        ],
    }
    model_hashes = {}
    for label in ("M1-clean1000", "M2-clean1000", "M1-expanded1149", "M2-expanded1149"):
        method = label.split("-")[0]
        model_meta = previous_meta["models"][label]
        checkpoint = Path(model_meta["checkpoint"])
        require(checkpoint.resolve().is_relative_to(previous.resolve()), "checkpoint outside pinned experiment")
        require(sha256(checkpoint) == model_meta["checkpoint_sha256"], "checkpoint hash mismatch")
        state = torch.load(checkpoint, map_location="cpu")
        require(state["method"] == method and state["encoder"] == builder.signature, "checkpoint signature differs")
        model = make_head(method, state["dimension"])
        model.load_state_dict(state["model"], strict=True)
        model.eval().requires_grad_(False)
        scores[label] = predict_scores(model, ds)
        model_hashes[label] = model_meta["checkpoint_sha256"]

    summary = {}
    for label, values in scores.items():
        result = publish_scores(ds, label.split("-")[0], label, values, output, {"run_id": run_id})
        old = read_json(previous / "predicted" / f"{label}-evaluation.json")
        summary[label] = {"u50000": result["summary"], "u18000": old["summary"]}
        print(label, result["summary"], flush=True)

    b0_dir = output / "B0"
    run_b0(paths["candidates"], paths["queries"], paths["lexicon"], b0_dir, run_id + "-predicted-B0")
    result = evaluate_files(paths["queries"], b0_dir / "predictions.jsonl", paths["candidates"],
                            paths["annotations"], b0_dir / "evaluation.json")
    summary["B0"] = {"u50000": result["summary"]}
    require(all(v["u50000"]["queries"] == 642 for v in summary.values()), "evaluation denominator changed")
    write_json(output / "comparison.json", summary)
    write_json(output / "B2-coverage.json", rule_coverage(pack, ds["queries"], lexicon, weight))
    metadata.update(
        status="completed", candidate_set_id=config["candidate_set_id"], image_manifest_sha256=ds["hashes"]["images"],
        model_checkpoints_sha256=model_hashes, reference_experiment=str(previous),
        queries=len(ds["queries"]), images=len(pack.rows),
        candidate_count=sum(len(row["candidates"]) for row in pack.rows.values()),
        encoded_visual_images=builder.encoded_images, encoded_texts=builder.encoded_texts,
        prior_cache_hits=builder.hits,
    )
    write_json(output / "run_metadata.json", metadata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    run(args.config, args.output, args.run_id)
