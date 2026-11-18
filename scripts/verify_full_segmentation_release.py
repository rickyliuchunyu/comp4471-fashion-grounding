"""Verify release, parent retention and protection metadata for the full Fashionpedia increment."""
import argparse
from collections import Counter
from pathlib import Path

from accept_segmentation_expansion import check_isolation, check_subsets, indexed
from fashion_grounding.data.schema import read_json, require, sha256, write_json


def pinned(path, digest):
    require(sha256(path) == digest, f"checksum differs: {path}")
    return read_json(path)


def fingerprint_key(row):
    split = {"official_train2020": "train", "official_val2020": "val"}[row["source_split"]]
    return f"{split}-{row['source_image_id']}"


def verify(release, previous, post):
    root = Path(release["path"])
    delta = Path(release["added_cpu_view"])
    parent = root.parent / release["parent"]
    catalog = pinned(root / "SHA256SUMS.json", release["checksums_sha256"])
    descriptor = pinned(root / "RELEASE.json", release["release_sha256"])
    require(descriptor == {key: release[key] for key in descriptor}, "release descriptor")
    require(root.name == descriptor["release_id"] and str(parent) == previous["package"], "lineage")

    def released(name):
        return pinned(root / name, catalog[name])

    old = {}
    for name, digest in descriptor["parent_pins"].items():
        if name in previous["package_pins"]:
            require(digest == previous["package_pins"][name], "prior acceptance pin differs")
        old[name] = pinned(parent / name, digest)
    old_assets = {p: h for p, h in old["SHA256SUMS.json"].items() if p.startswith(("images/", "masks/"))}
    require(all(catalog.get(p) == h for p, h in old_assets.items()), "old image/mask digest changed")
    require(catalog["mapping_config.json"] == descriptor["parent_pins"]["mapping_config.json"], "old mapping")
    old_members = released("old_subset_membership.json")
    new_members = released("added_subset_membership.json")
    full_manifest = pinned(root / "image_manifest.json", release["manifest_sha256"])
    full_coco = pinned(root / "annotations_coco.json", release["annotations_sha256"])
    fidelity = released("audit/source_fidelity.json")
    require(catalog["audit/source_fidelity.json"] == release["source_fidelity_sha256"]
            and fidelity["source_sha256"] == release["source_annotation_sha256"], "source audit binding")
    added_manifest = pinned(delta / "image_manifest.json", fidelity["delta_manifest_sha256"])
    added_coco = pinned(delta / "annotations_coco.json", fidelity["delta_annotation_sha256"])
    fields = [
        (old["image_manifest.json"], full_manifest, added_manifest, "images", "image_id", "image_ids"),
        (old["annotations_coco.json"], full_coco, added_coco, "images", "id", "image_ids"),
        (old["annotations_coco.json"], full_coco, added_coco, "annotations", "id", "annotation_ids"),
    ]
    for old_doc, full_doc, added_doc, field, key, member in fields:
        rows = check_subsets(old_doc[field], full_doc[field], old_members[member], new_members[member], key)
        require(indexed(rows, key) == indexed(added_doc[field], key), f"CPU view differs: {field}")
    require(full_coco["categories"] == old["annotations_coco.json"]["categories"] == added_coco["categories"], "categories")
    added = added_manifest["images"]
    added_keys = {fingerprint_key(r) for r in added}
    require(all(r["source_dataset"] == "fashionpedia" and r["split"] == "train"
                and r["source_split"] == "official_train2020" and r["image_id"] == int(r["source_image_id"]) + 1
                and catalog[r["file_name"]] == r["sha256"] for r in added), "added source/asset identity")
    eligible = indexed(released("audit/eligible.json"), "key")
    require(set(eligible) == added_keys and all(
        (eligible[fingerprint_key(r)]["group_id"], eligible[fingerprint_key(r)]["sha256"]) == (r["group_id"], r["sha256"])
        for r in added), "added fingerprint audit does not cover manifest")

    inputs = released("audit/protection_inputs.json")
    require(all(sha256(p) == h for p, h in inputs.items()), "protection input changed")
    frozen = indexed(released("audit/protected_fingerprints.json"), "key")
    protected = {"parent": old["image_manifest.json"]["images"]}
    for name, directory in previous["protected"].items():
        path = Path(directory) / "image_manifest.json"
        require(inputs[str(path)] == previous["protected_manifest_pins"][name], f"{name} manifest pin")
        protected[name] = read_json(path)["images"]
    for rows in protected.values():
        for row in rows:
            fp = frozen[fingerprint_key(row)]
            require((fp["sha256"], fp["group_id"]) == (row["sha256"], row["group_id"]), "frozen identity")
    registry_paths = [p for p in inputs if Path(p).name == "dev_exposure_registry.json"]
    require(len(registry_paths) == 1 and inputs[registry_paths[0]] == release["dev_exposure_registry_sha256"], "exposure registry pin")
    exposures = indexed(read_json(registry_paths[0])["images"], "image_id")
    exposed = []
    for image_id in exposures:
        source_id = image_id - 1_000_000_001  # Project validation ID namespace.
        fp = frozen[f"val-{source_id}"]
        exposed.append(dict(image_id=image_id, source_split="official_val2020", source_dataset="fashionpedia",
                            source_image_id=str(source_id), sha256=fp["sha256"], group_id=fp["group_id"]))
    require(len(exposed) == 50, "exposure registry scope")
    protected["exposed"] = exposed
    overlaps = check_isolation(added, protected)
    require(added_keys.isdisjoint(frozen)
            and {r["group_id"] for r in added}.isdisjoint(r["group_id"] for r in frozen.values())
            and {r["sha256"] for r in added}.isdisjoint(r["sha256"] for r in frozen.values()), "reserved fingerprint overlap")
    require(post["status"] == "passed" and post["checksums_sha256"] == release["checksums_sha256"]
            and post["retained_duplicate_flags"] == 0 and post["protected_fingerprints"] == len(frozen)
            and post["parent_and_delta_union_verified"], "published duplicate audit binding")

    ledger = released("audit/ledger.json")
    old_keys = {fingerprint_key(r) for r in protected["parent"]}
    for disposition, expected in (("added", added_keys), ("parent", old_keys)):
        require({"train-" + k for k, reason in ledger.items() if reason == disposition} == expected, "source ledger")
    account = dict(Counter(ledger.values()))
    require(account == descriptor["source_account"] and account == released("audit/counts.json"), "source totals")
    bridge = released("audit/frozen_bridge_clues.json")
    excluded = released("audit/duplicates.json")["excluded"]
    partitions = {k: {fingerprint_key(r) for r in rows} for k, rows in protected.items()}
    components = []
    all_quarantined = set()
    for component in bridge["components"]:
        keys = set(component)
        quarantined = {r["key"] for r in excluded if r["reason"] == "component_reaches_frozen_or_protected"
                       and set(r["component"]) == keys}
        hits = {k: len(keys & other) for k, other in partitions.items()}
        require(hits["parent"] and any(hits[k] for k in ("dev", "test", "exposed")), "bridge protected anchors")
        require(keys.isdisjoint(added_keys) and quarantined == keys.difference(frozen)
                and all(ledger[k.removeprefix("train-")] == "component_reaches_frozen_or_protected"
                        for k in quarantined), "bridge quarantine incomplete")
        all_quarantined.update(quarantined)
        components.append({"members": len(keys), "frozen_hits": hits, "quarantined_new": len(quarantined)})
    require(len(components) == bridge["count"] and len(all_quarantined) == bridge["quarantined_new_source_images"], "bridge totals")
    full = {"images": len(full_manifest["images"]), "groups": len({r["group_id"] for r in full_manifest["images"]}),
            "instances": len(full_coco["annotations"])}
    require(all(n == release[k] for k, n in full.items()), "release totals")
    require(fidelity["verified_new_images"] == post["retained_new_images"] == len(added) == release["added_images"]
            and fidelity["verified_new_masks"] == post["all_new_legal_instances"] == len(added_coco["annotations"])
            == release["added_instances"], "added audit totals")
    return {"status": "passed", "release_id": root.name, "release_sha256": release["release_sha256"], "totals": full,
            "old_images_exact": len(old_members["image_ids"]), "old_annotations_exact": len(old_members["annotation_ids"]),
            "old_asset_digests_equal": len(old_assets), "source_account": account, "added_intersections": overlaps,
            "protected_fingerprints": len(frozen), "producer_retained_duplicate_flags": post["retained_duplicate_flags"],
            "exposure_registry_sha256": release["dev_exposure_registry_sha256"],
            "exposure_status_counts": dict(Counter(r["status"] for r in exposures.values())),
            "historical_components": components, "quarantined_new_in_historical_components": len(all_quarantined),
            "historical_review_status": bridge["review_status"], "formal_freeze": False,
            "reused_source_pixel_checks": fidelity["verified_new_masks"], "test_access": "identity metadata only"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("published", "previous", "postpublish", "output"):
        parser.add_argument(arg, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "receipt already exists")
    write_json(args.output, verify(read_json(args.published), read_json(args.previous), read_json(args.postpublish)))
