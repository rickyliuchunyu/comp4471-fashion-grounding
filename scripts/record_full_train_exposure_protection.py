"""Bind the two exposure lists and the producer's bridge decision to the accepted full release.

Run in an empty output directory with release_check.json, published.json and
final_checkpoint.json from the segmentation handoff. Reads only metadata.
"""
import argparse
from collections import Counter
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import subprocess


PRODUCER = "07b0125a14db02e9b448c097d40df491c70d2c7d"
LISTS = {
    "challenge": ("language_targeted_20261117", "dev_challenge_draft", 39, 11),
    "direction_pairs": ("direction_pairs_20261118", "dev_pair_draft", 74, 22),
}
QUARANTINE = "dev_exposed_quality_quarantine"


def check(condition, message):
    if not condition:
        raise ValueError(message)


def index(rows, key):
    result = {row[key]: row for row in rows}
    check(len(result) == len(rows), f"duplicate {key}")
    return result


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf8")


def verify(base, producer_repo, out):
    pins = {}
    def decode(key, data, expected=None):
        digest = sha256(data).hexdigest()
        check(expected is None or digest == expected, f"changed input: {key}")
        pins[key] = digest
        return json.loads(data)

    def server(path, expected=None):
        return decode(str(path.relative_to(base)), path.read_bytes(), expected)

    def upstream(name, expected=None):
        path = "docs/handoffs/data_eval/" + name
        raw = subprocess.check_output(["git", "-C", str(producer_repo), "show", PRODUCER + ":" + path])
        return decode(path, raw, expected)

    def consumer(name, expected=None):
        return decode(name, (out / name).read_bytes(), expected)

    published = consumer("published.json")
    full = base / "shared/processed" / published["release_id"]
    release = server(full / "RELEASE.json", published["release_sha256"])
    catalog = server(full / "SHA256SUMS.json", published["checksums_sha256"])
    manifest = server(full / "image_manifest.json", published["manifest_sha256"])["images"]
    guard_inputs = server(full / "audit/protection_inputs.json", catalog["audit/protection_inputs.json"])
    old_fingerprints = index(server(full / "audit/protected_fingerprints.json", catalog["audit/protected_fingerprints.json"]), "key")
    test = full.parent / "fp_test400_holdout_v1" / "image_manifest.json"
    holdout = server(test, guard_inputs[str(test)])["images"]

    decision = upstream("full_train_20261118_evidence/historical_bridge_adjudication.json")
    prior = consumer("release_check.json", decision["consumer_receipt_sha256"])
    clues = upstream("full_train_20261118_evidence/frozen_bridge_clues.json", decision["clues_sha256"])
    checkpoint = consumer("final_checkpoint.json")
    check(decision["release"] == release["release_id"] and decision["release_file_hashes"]["RELEASE.json"]
          == prior["release_sha256"] == published["release_sha256"], "decision release")
    check(decision["checkpoint_sha256"] == checkpoint["checkpoint_sha256"] and not decision["member_id_changes"]
          and decision["checkpoint_decision"] == "eligible_as_model_initialization_for_this_duplicate_review", "decision scope")
    groups = []
    for i, (row, component, accepted) in enumerate(zip(decision["groups"], clues["components"],
                                                       prior["historical_components"], strict=True), 1):
        endpoints = index(row["endpoints"], "source_key")
        parents = {k for k, v in endpoints.items() if v["partition"] == "parent"}
        pairs = {(x["parent"], x["protected"]) for x in row["direct_comparisons"]}
        check(row["group"] == i and pairs == set(product(parents, endpoints.keys() - parents))
              and len(pairs) == len(row["direct_comparisons"]), "endpoint comparison coverage")
        check(set(endpoints) <= set(component) and len(component) == row["members_in_original_clue"]
              and len(set(component) - endpoints.keys() - set(row["frozen_auxiliary_keys"]))
              == row["quarantined_new"] == accepted["quarantined_new"], "historical group scope")
        check(row["disposition"] == "retain_existing_members_and_keep_new_quarantined"
              and row["confirmed_cross_split_source_matches"] == 0, "changed disposition")
        groups.append(dict(group=i, comparisons=len(pairs), quarantine=row["quarantined_new"]))
    check(len(groups) == 5 and sum(g["quarantine"] for g in groups) == 123, "historical group count")

    run = base / "runs/data_eval/20261118-nonshoe-pairs-v1"
    guard = server(run / "dev_pool_guard.json")
    separation = upstream("direction_pairs_20261118_evidence/separation.json")
    check(guard["protected"] == separation["guard_protected"] == len(old_fingerprints) + published["added_images"],
          "guard scope")
    pool = index(server(run / "dev_source/image_manifest.json")["images"], "image_id")
    safe_keys = set(guard["eligible"])
    check({"val-" + str(r["source_image_id"]) for r in pool.values()} == safe_keys
          and len(pool) == len(safe_keys) == separation["new_candidate_pool_images"], "guarded pool membership")
    screen = index(server(run / "dev-visual-screen.json"), "image_id")
    preflight = server(full / "audit/input_preflight.json", catalog["audit/input_preflight.json"])
    inventory = base / "runs/data_eval/20261112-large-data-regrouped/fingerprints.json"
    fingerprints = index(server(inventory, preflight["input_hashes"][str(inventory)]), "key")

    records = {}
    sources = {}
    for batch, (handoff, keep_status, kept_n, rejected_n) in LISTS.items():
        rel = handoff + "_evidence/dev_exposure_registry.json"
        original = batch == "challenge"
        rows = index(upstream(rel, release["dev_exposure_registry_sha256"] if original else None)["images"], "image_id")
        counts = Counter(r["status"] for r in rows.values())
        check(counts == {keep_status: kept_n, QUARANTINE: rejected_n} and records.keys().isdisjoint(rows), "registry membership")
        kept = {i for i, row in rows.items() if row["status"] == keep_status}
        pkg = "fp_language_dev39_challenge_draft_v1" if original else "fp_language_direction_pairs_dev_v1"
        path = full.parent / pkg / "image_manifest.json"
        kept_rows = index(server(path, guard_inputs[str(path)] if original else None)["images"], "image_id")
        check(kept == kept_rows.keys(), "kept manifest membership")
        if not original:
            check(rows.keys() == screen.keys() and all(screen[i]["retained"] is (i in kept) for i in rows)
                  and rows.keys() <= pool.keys(), "screened vs protected IDs")
        for image_id, row in rows.items():
            source_id = image_id - 1_000_000_001
            key = f"val-{source_id}"
            fp = (old_fingerprints if original else fingerprints)[key]
            check(fp["source_split"] == "val" and fp["source_image"]["id"] == source_id, "fingerprint identity")
            if not original or image_id in kept:
                im = (kept_rows if original else pool)[image_id]
                check(im["source_dataset"] == "fashionpedia" and im["source_split"] == "official_val2020"
                      and str(im["source_image_id"]) == str(source_id) and im["sha256"] == fp["sha256"]
                      and im["group_id"] == fp["group_id"], "manifest/fingerprint mismatch")
                if not original and image_id in kept:
                    check(all(kept_rows[image_id][k] == im[k] for k in ("source_image_id", "source_split", "group_id", "sha256")),
                          "draft asset differs from guarded pool")
            records[image_id] = dict(image_id=image_id, source_key=key, group_id=fp["group_id"], sha256=fp["sha256"],
                                     batch=batch, status=row["status"])
        sources[batch] = dict(path="docs/handoffs/data_eval/" + rel, sha256=pins["docs/handoffs/data_eval/" + rel],
                              counts=dict(counts))
    check(len(records) == 146, "expected 146 distinct exposures")
    overlaps = {}
    for name, rows in (("full_train", manifest), ("test400", holdout)):
        overlaps[name] = {field: len({r[field] for r in rows} & {r[field] for r in records.values()})
                          for field in ("image_id", "group_id", "sha256")}
        splits = {"official_train2020": "train", "official_val2020": "val"}
        source_keys = {splits[r["source_split"]] + "-" + str(r["source_image_id"]) for r in rows}
        overlaps[name]["source_key"] = len(source_keys & {r["source_key"] for r in records.values()})
        check(not any(overlaps[name].values()), f"exposure overlap: {name}")
    registry = dict(version="segmentation-dev-exposures-20261118-v1", release=release["release_id"],
                    release_sha256=published["release_sha256"], producer_commit=PRODUCER, source_registries=sources,
                    source_dataset="fashionpedia", source_split="official_val2020", forbidden_splits=["train", "test"],
                    distinct_images=len(records), images=[records[i] for i in sorted(records)])
    registry_path = out / "protected_exposures.json"
    save(registry_path, registry)
    save(out / "verification.json", dict(registry_sha256=sha256(registry_path.read_bytes()).hexdigest(),
         full_train_images=len(manifest), test_images=len(holdout), overlaps=overlaps, screened_from_guarded_pool=len(screen),
         guarded_pool=len(pool), guard_protected=guard["protected"], historical_groups=groups,
         review_scope=decision["review_scope"], checkpoint_sha256=decision["checkpoint_sha256"],
         test_images_seen_by_producer_for_dedup_only=decision["protected_test_images_viewed_for_dedup_only"],
         input_sha256=pins))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("server_base", type=Path)
    parser.add_argument("producer_repo", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    verify(args.server_base, args.producer_repo, args.output)
