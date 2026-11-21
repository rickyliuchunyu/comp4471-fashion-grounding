"""Replay frozen M2 scores with the previously selected B2 position bonus on reviewed dev."""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from evaluate_predictions import evaluate_files
from fashion_grounding.data.schema import sha256


def read_rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s]


def run(config, reference, output):
    cfg = json.loads(config.read_text())
    paths = {k: Path(cfg[k]) for k in ("queries", "annotations", "candidates")}
    assert all(sha256(path) == cfg["sha256"][k] for k, path in paths.items())
    metadata = json.loads((reference / "run_metadata.json").read_text())
    assert metadata["input_sha256"] == cfg["sha256"]
    assert metadata["candidate_set_id"] == cfg["candidate_set_id"]
    ref = reference / "predicted"
    names = ("M2-clean1000", "B1", "B2-selected")
    scores = [read_rows(ref / (name + "-scores.jsonl")) for name in names]
    predictions = read_rows(ref / "M2-clean1000-predictions.jsonl")
    assert all(p["split"] == "dev" for p in predictions)
    assert len(set(p["query_id"] for p in predictions)) == len(predictions)
    for p, learned, base, rule in zip(predictions, *scores, strict=True):
        assert all(row["query_id"] == p["query_id"] for row in (learned, base, rule))
        ids = learned["candidate_ids"]
        assert ids == base["candidate_ids"] == rule["candidate_ids"]
        p.update(method_id="M2+B2-position", run_id=output.parent.name + "-reviewed")
        if not ids:
            assert p["status"] != "success"
            continue
        assert p["status"] == "success"
        m, b, r = (np.asarray(row["scores"]) for row in (learned, base, rule))
        assert all(len(v) == len(ids) and np.isfinite(v).all() for v in (m, b, r))
        assert ids[int(np.argmax(m))] == p["selected_candidate"]["candidate_id"]
        hybrid = m + (r - b)
        best = int(np.argmax(hybrid))
        p["selected_candidate"] = {"candidate_id": ids[best], "grounding_score": float(hybrid[best])}
    output.mkdir(parents=True, exist_ok=False)
    (output / "inputs").mkdir()
    shutil.copyfile(paths["candidates"], output / "inputs/candidates_predicted.json")
    prediction_path = output / "predictions.jsonl"
    prediction_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions))
    result = evaluate_files(paths["queries"], prediction_path, paths["candidates"], paths["annotations"],
                            output / "evaluation.json")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "reference", "output"):
        p.add_argument("--" + name, required=True, type=Path)
    args = p.parse_args()
    run(args.config, args.reference, args.output)
