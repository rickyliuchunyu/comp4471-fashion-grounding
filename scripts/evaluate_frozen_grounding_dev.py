"""Evaluate fixed grounding heads with an explicitly pinned dev candidate pack."""
import argparse
import subprocess
import sys
from pathlib import Path

import torch

from fashion_grounding.data.schema import read_json, require, sha256, write_json
from fashion_grounding.grounding.binding import binding_geometry
from fashion_grounding.grounding.cache import FeatureCache
from fashion_grounding.grounding.controlled import FeatureBuilder, predict_scores, publish_scores
from fashion_grounding.grounding.inputs import CandidatePack
from fashion_grounding.grounding.rules import position_scores, rule_coverage, shared_parse
from fashion_grounding.grounding.spatial import make_head
from scripts.evaluate_predictions import evaluate_files
from scripts.run_b0 import run as run_b0


def check_release(config, paths, pack):
    """The release receipt freezes an immutable offline threshold-trial package."""
    receipt = config.get("release")
    if receipt is None:
        return
    path = Path(receipt["path"])
    require(sha256(path) == receipt["sha256"], "release receipt changed")
    released = read_json(path)
    selected = released["selected_package"]
    expected = config["checkpoint_sha256"]
    require(Path(released["selected_entry"]) == paths["candidates"]
            and Path(selected["path"]).resolve() == pack.path
            and released["status"] == "available_fixed_for_development"
            and selected["sha256"] == pack.digest
            and selected["candidate_set_id"] == pack.package["candidate_set_id"]
            and released["queries_sha256"] == config["sha256"]["queries"]
            and released["checkpoint_sha256"] == expected == pack.package["provenance"]["checkpoint_sha256"]
            and released["selected_threshold"] == config["threshold"] == pack.package["provenance"]["score_threshold"],
            "package differs from the frozen release")
    require(len(pack.rows) == config["image_count"]
            and sum(len(row["candidates"]) for row in pack.rows.values()) == config["candidate_count"],
            "wrong frozen candidate counts")


def previous_results(config, label, previous, hashes):
    if "reference_evaluations" in config:
        spec = config["reference_evaluations"].get(label)
        if spec is None:
            return {}
        result = read_json(spec["path"])
        identity = result["provenance"]
        for key in ("queries", "annotations"):
            require(identity[key + "_sha256"] == hashes[key], "comparison uses different " + key)
        require(identity["candidates_sha256"] == spec["candidate_sha256"], "wrong comparison candidate package")
        return {spec["id"]: result["summary"]}
    if label == "B0":
        return {}
    return {"u18000": read_json(previous / "predicted" / f"{label}-evaluation.json")["summary"]}


def run(config_path, output, run_id, candidates=None):
    config = read_json(config_path)
    paths = {key: Path(config[key]) for key in ("candidates", "queries", "annotations", "lexicon")}
    if candidates is not None:
        require(candidates.resolve() == paths["candidates"].resolve(), "--candidates differs from the pinned configuration")
        paths["candidates"] = candidates
    hashes = {key: sha256(path) for key, path in paths.items()}
    require(hashes == config["sha256"], "candidate, query, annotation or lexicon bytes changed")
    previous = Path(config["previous_experiment"])
    selection = read_json(previous / "B2-dev-selection.json")
    require(selection["query_sha256"] == hashes["queries"], "B2 selection used a different query snapshot")
    require(selection["lexicon_sha256"] == hashes["lexicon"], "B2 lexicon changed since selection")
    weight = selection["selected_weight"]
    pack = CandidatePack(paths["candidates"])
    require(pack.package["candidate_source"] == "predicted"
            and pack.package["candidate_set_id"] == config["candidate_set_id"], "candidate set mismatch")
    check_release(config, paths, pack)

    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    builder = FeatureBuilder(config, output)
    reference_cache = Path(config.get("text_feature_cache", previous / "features"))
    require(reference_cache.is_dir(), "missing reference feature cache")
    builder.readonly = FeatureCache(reference_cache)

    heads = config.get("heads")
    if heads is None:
        models = read_json(previous / "run_metadata.json")["models"]
        heads = {label: models[label] for label in ("M1-clean1000", "M2-clean1000", "M1-expanded1149", "M2-expanded1149")}
    use_binding = any(head["method"] == "M2-binding" for head in heads.values())
    if use_binding:
        builder.candidate_features = binding_geometry

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
        "release": config.get("release"),
    }
    write_json(output / "run_metadata.json", metadata)
    dataset = builder.build(config["candidate_set_id"], paths["candidates"], paths["queries"], paths["annotations"])
    require(len(pack.rows) == 250 and len(dataset["queries"]) == 642, "unexpected scope")
    lexicon = read_json(paths["lexicon"])
    scores = {
        "B1": ("B1", [ex["base"] if ex["candidates"] else None for ex in dataset["examples"]]),
        "B2-selected": ("B2", [
            ex["base"] + weight * position_scores(
                ex["candidates"], shared_parse(ex["query"].query, ex["query"].language, lexicon)
            ) if ex["candidates"] else None
            for ex in dataset["examples"]
        ]),
    }
    model_hashes = {}
    for label, info in heads.items():
        method = info["method"]
        checkpoint = Path(info["checkpoint"])
        require(sha256(checkpoint) == info["checkpoint_sha256"], "checkpoint hash mismatch")
        state = torch.load(checkpoint, map_location="cpu")
        require(state["method"] == method and state["encoder"] == builder.signature, "checkpoint signature differs")
        model = make_head(method, state["dimension"])
        model.load_state_dict(state["model"], strict=True)
        model.eval().requires_grad_(False)
        view = dataset
        if use_binding and method != "M2-binding":
            view = {**dataset, "examples": [{**ex, "geometry": ex["geometry"][:, :12].copy()}
                                            for ex in dataset["examples"]]}
        scores[label] = ("M2" if method == "M2-binding" else method, predict_scores(model, view))
        model_hashes[label] = info["checkpoint_sha256"]

    summary = {}
    evaluation_id = config.get("evaluation_id", "u50000")
    expected_coverage = config.get("matched_coverage_count")
    for label, (method, values) in scores.items():
        result = publish_scores(dataset, method, label, values, output, {"run_id": run_id})
        if expected_coverage is not None:
            require(sum(row["covered"] for row in result["rows"]) == expected_coverage, "frozen coverage count differs")
        summary[label] = {evaluation_id: result["summary"], **previous_results(config, label, previous, hashes)}
        print(label, result["summary"], flush=True)

    if config.get("include_B0", True):
        b0_dir = output / "B0"
        run_b0(paths["candidates"], paths["queries"], paths["lexicon"], b0_dir, run_id + "-predicted-B0")
        result = evaluate_files(paths["queries"], b0_dir / "predictions.jsonl", paths["candidates"],
                                paths["annotations"], b0_dir / "evaluation.json")
        summary["B0"] = {evaluation_id: result["summary"], **previous_results(config, "B0", previous, hashes)}
    require(all(v[evaluation_id]["queries"] == 642 for v in summary.values()), "evaluation denominator changed")
    write_json(output / "comparison.json", summary)
    write_json(output / "B2-coverage.json", rule_coverage(pack, dataset["queries"], lexicon, weight))
    if config.get("release"):
        require(builder.encoded_images == len({q.image_id for q in dataset["queries"]}), "new release visual features were not all rebuilt")
    metadata.update(
        status="completed", candidate_set_id=config["candidate_set_id"], image_manifest_sha256=dataset["hashes"]["images"],
        checkpoint_sha256=pack.package["provenance"]["checkpoint_sha256"],
        threshold=pack.package["provenance"].get("score_threshold"),
        model_checkpoints_sha256=model_hashes, reference_experiment=str(previous),
        queries=len(dataset["queries"]), images=len(pack.rows),
        candidate_count=sum(len(row["candidates"]) for row in pack.rows.values()),
        encoded_visual_images=builder.encoded_images, encoded_texts=builder.encoded_texts,
        prior_cache_hits=builder.hits,
    )
    write_json(output / "run_metadata.json", metadata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    run(args.config, args.output, args.run_id, args.candidates)
