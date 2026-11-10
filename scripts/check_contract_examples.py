"""Check synthetic contract fixtures only; no model, data conversion or evaluation."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def referenced(parent, name, expected_hash=None):
    relative = Path(name)
    require(not relative.is_absolute() and ".." not in relative.parts,
            f"Unsafe relative path: {name}")
    path = parent / relative
    require(path.is_file(), f"Missing reference: {path}")
    if expected_hash is not None:
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        require(actual == expected_hash, f"Hash mismatch: {path}")
    return path


def common(obj):
    require(obj.get("schema_version") == 3, "Expected contract schema 3")
    require(obj.get("illustrative_only") is True, "Fixtures must stay illustrative")


def check(root):
    examples = root / "examples"
    manifest = read(examples / "image_manifest.json")
    common(manifest)
    images = {}
    for image in manifest["images"]:
        iid = image["image_id"]
        require(type(iid) is int and iid > 0 and iid not in images, "Bad image ID")
        require(image["split"] == "fixture", "Synthetic image must use fixture split")
        path = referenced(examples, image["file_name"], image["sha256"])
        tokens = path.read_text().split()
        require(tokens[0] == "P3", "This checker supports the fixture P3 image only")
        w, h, maximum = map(int, tokens[1:4])
        require((w, h, maximum) == (image["width"], image["height"], 255), "Image size")
        values = list(map(int, tokens[4:]))
        require(len(values) == 3 * w * h and all(0 <= v <= 255 for v in values),
                "Invalid RGB image payload")
        images[iid] = image

    packages = {}
    for filename in ("candidates_gt.json", "candidates_predicted.json", "candidates_empty.json"):
        package = read(examples / filename)
        common(package)
        ref = package["image_manifest"]
        require(referenced(examples, ref["path"], ref["sha256"]).name == "image_manifest.json",
                "Unexpected image manifest")
        ids, image_ids, annotation_ids = set(), set(), set()
        for item in package["images"]:
            iid = item["image_id"]
            require(iid in images and iid not in image_ids, "Candidate image ID mismatch")
            image_ids.add(iid)
            w, h = images[iid]["width"], images[iid]["height"]
            require(item["image_size_wh"] == [w, h], "Candidate image dimensions")
            candidates = item["candidates"]
            require(item["status"] == ("ok" if candidates else "no_candidates"), "Candidate status")
            decoder_ids = set()
            for candidate in candidates:
                cid = candidate["candidate_id"]
                require(isinstance(cid, str) and cid and cid not in ids, "Duplicate/invalid candidate ID")
                ids.add(cid)
                require(type(candidate["category_id"]) is int and 1 <= candidate["category_id"] <= 8,
                        "Invalid project category")
                mask_path = referenced(examples, candidate["mask_reference"], candidate["mask_sha256"])
                mask = read(mask_path)
                require(mask["size"] == [h, w], "Mask dimensions")
                counts = mask["counts"]
                require(isinstance(counts, list) and counts and
                        all(type(n) is int and n >= 0 for n in counts) and sum(counts) == w*h,
                        "Invalid uncompressed RLE counts")
                pixels, bit = [], 0
                for count in counts:
                    pixels.extend([bit] * count)
                    bit = 1 - bit
                foreground = [(i // h, i % h) for i, v in enumerate(pixels) if v]
                require(foreground, "Empty candidate mask")
                xs, ys = zip(*foreground)
                bbox = [min(xs), min(ys), max(xs)+1, max(ys)+1]
                require(candidate["bbox_xyxy"] == bbox, "Mask/bbox mismatch")
                if package["candidate_source"] == "gt":
                    aid = candidate["source_annotation_id"]
                    require(type(aid) is int and aid > 0 and aid not in annotation_ids, "Bad GT annotation ID")
                    annotation_ids.add(aid)
                    require(candidate["score"] is None and candidate["decoder_query_id"] is None,
                            "GT must not claim model scores/query IDs")
                else:
                    score, decoder = candidate["score"], candidate["decoder_query_id"]
                    require(type(score) in (int, float) and math.isfinite(score), "Invalid score")
                    require(type(decoder) is int and decoder >= 0 and decoder not in decoder_ids,
                            "Duplicate/invalid decoder query")
                    decoder_ids.add(decoder)
                    require(candidate["source_annotation_id"] is None, "Predicted candidate leaks GT ID")
            key = (lambda c: c["source_annotation_id"]) if package["candidate_source"] == "gt" else (
                lambda c: (-c["score"], c["candidate_id"]))
            require(candidates == sorted(candidates, key=key), "Candidate order mismatch")
        require(image_ids == set(images), "Candidate package misses manifest image")
        require(package["candidate_set_id"] not in packages, "Duplicate candidate set")
        packages[package["candidate_set_id"]] = package

    query = read(examples / "query_record.json")
    common(query)
    require(query["query_id"] and query["image_id"] in images, "Query identifiers")
    require(query["split"] == images[query["image_id"]]["split"], "Query split mismatch")
    gt = read(examples / "candidates_gt.json")["images"][0]["candidates"]
    target = [c for c in gt if c["source_annotation_id"] == query["target_instance_id"]]
    require(len(target) == 1 and target[0]["category_id"] == query["target_category_id"], "Query target mismatch")
    require(query["annotator"] != query["reviewer"] and query["review_status"] != "accepted",
            "Synthetic query cannot claim accepted annotation")
    seen = set()
    for filename in ("prediction.json", "prediction_no_candidates.json", "prediction_inference_error.json"):
        prediction = read(examples / filename)
        common(prediction)
        for field in ("query_id", "image_id", "query", "language", "split"):
            require(prediction.get(field) == query[field], f"Prediction {field} mismatch")
        key = (prediction["run_id"], prediction["query_id"])
        require(key not in seen, "Duplicate run/query prediction")
        seen.add(key)
        ref = prediction["candidate_manifest"]
        package = read(referenced(examples, ref["path"], ref["sha256"]))
        require(prediction["candidate_set_id"] == package["candidate_set_id"] and
                prediction["candidate_source"] == package["candidate_source"], "Wrong candidate provenance")
        item = next(i for i in package["images"] if i["image_id"] == prediction["image_id"])
        selected, status = prediction["selected_candidate"], prediction["status"]
        if status == "success":
            require(selected is not None and any(c["candidate_id"] == selected["candidate_id"]
                    for c in item["candidates"]), "Selected candidate missing from same image")
            require(prediction["error"] is None, "Success with error")
        else:
            require(selected is None, "Failure must not select a candidate")
            require(status in ("no_candidates", "inference_error"), "Unexpected fixture failure")
            if status == "no_candidates":
                require(not item["candidates"] and item["status"] == status and prediction["error"] is None,
                        "Invalid no-candidates record")
            else:
                require(prediction["error"] and prediction["error"]["code"] and prediction["error"]["message"],
                        "Error details missing")

    rules = read(root / "configs/rule_parser.json")
    require({a["category_id"] for a in rules["category_aliases"]} == set(range(1, 9)), "Rule categories")
    for language in ("en", "zh"):
        aliases = [word for a in rules["category_aliases"] for word in a[language]]
        require(len(aliases) == len(set(aliases)), "Ambiguous duplicate aliases")
    plan = read(root / "configs/training_plan.json")
    require(plan["grounding"]["shared_parser_config"] == "configs/rule_parser.json", "Parser config drift")
    require(plan["candidates"]["threshold_selection"]["selected_threshold"] is None,
            "No candidate threshold selection has been performed")
    with (root / "experiments/registry.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    require(len({row["experiment_id"] for row in rows if row["record_type"] == "plan"}) == 14,
            "Missing/duplicate planned experiments")
    for row in rows:
        require(None not in row and all(v is not None for v in row.values()), "CSV column mismatch")
        if row["record_type"] == "plan":
            empty = ["run_id", "executed_by", "mask_ap", "target_selection_accuracy",
                     "oracle_selection_accuracy", "predicted_selection_accuracy", "mean_target_mask_iou",
                     "mean_latency_ms", "p95_latency_ms", "success_at_0_5"]
            require(all(row[k] == "" for k in empty), "Plan row claims execution or results")
            require(row["status"] in ("planned", "optional_planned"), "Plan status mismatch")
    print("PASS: synthetic image/RLE, IDs, references/hashes, candidate order, query/prediction states, rules and plan registry")
    print("Scope: fixtures only; no real dataset, model, GPU, shared parser or evaluator verified.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        check(args.root)
    except (ValueError, KeyError, TypeError, OSError, StopIteration) as exc:
        parser.exit(1, f"FAIL: {exc}\n")
