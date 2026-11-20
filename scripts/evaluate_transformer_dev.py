"""Exercise the public grounding command with pinned development inputs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from fashion_grounding.data.schema import read_json, require, sha256, write_json
from scripts.evaluate_predictions import evaluate_files
from scripts.run_mhp_binding_study import pair_metrics


def run(config_path, output, conditions, seed):
    config = read_json(config_path)
    release = Path(config["release"]["path"])
    require(sha256(release) == config["release"]["sha256"], "Transformer release changed")
    datasets = config["datasets"]
    selected = conditions or list(datasets)
    for name in selected:
        dataset = datasets[name]
        for key in ("candidates", "queries", "annotations"):
            require(sha256(dataset[key]) == dataset["sha256"][key], f"{name} {key} changed")

    output.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                       HF_HUB_OFFLINE="1", CUDA_VISIBLE_DEVICES="")
    summary = {"config_sha256": sha256(config_path), "status": "running", "results": {}}
    write_json(output / "summary.json", summary)
    for name in selected:
        dataset = datasets[name]
        queries = Path(dataset["queries"])
        candidates = Path(dataset["candidates"])
        draft = dataset["draft"]
        per_method = {}
        for method in dataset["methods"]:
            destination = output / name / method
            command = [
                sys.executable, "-m", "fashion_grounding.grounding.predict",
                "--method", method, "--transformer-release", str(release),
                "--candidates", str(candidates), "--queries", str(queries),
                "--cache", str(output / "features"), "--output", str(destination),
                "--run-id", f"{output.name}-{name}-{method}",
            ]
            for cache in config["reference_caches"]:
                command.extend(["--reference-cache", cache])
            if seed is not None:
                command.extend(["--transformer-seed", str(seed)])
            if draft:
                command.append("--draft-diagnostic")
            subprocess.run(command, check=True, env=environment)
            filename = "draft_predictions.jsonl" if draft else "predictions.jsonl"
            result = evaluate_files(queries, destination / filename, candidates, Path(dataset["annotations"]),
                                    destination / "evaluation.json", allow_draft=draft)
            rows = result["rows"]
            counts = {
                "queries": len(rows),
                "strict": sum(row["correct"] for row in rows),
                "mask_success": sum(row["success_at_0_5"] for row in rows),
                "covered": sum(row["covered"] for row in rows),
            }
            if draft:
                raw = [json.loads(line) for line in queries.read_text(encoding="utf-8").splitlines() if line.strip()]
                counts["pairs"] = pair_metrics({"raw": raw}, rows)
            per_method[method] = {**counts, "metrics": result["summary"],
                                  "provenance": result["provenance"]}
            print(name, method, counts, flush=True)
        summary["results"][name] = per_method
        write_json(output / "summary.json", summary)
    summary["status"] = "completed"
    write_json(output / "summary.json", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conditions", nargs="+", choices=("fp_combo", "mhp_gt", "mhp_combo"))
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    run(args.config, args.output, args.conditions, args.seed)
