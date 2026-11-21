"""Read only the published increment using the existing CPU segmentation adapter."""
import argparse
import inspect
import os
import time
from pathlib import Path

from fashion_grounding.data.schema import read_json, require, sha256, write_json
from fashion_grounding.segmentation.data import load_training_package


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("published", type=Path)
    parser.add_argument("fidelity", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    release = read_json(args.published)
    fidelity = read_json(args.fidelity)
    root = Path(release["path"])
    delta = Path(release["added_cpu_view"])
    require(not args.output.exists(), "receipt already exists")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU only")
    require(sha256(args.fidelity) == release["source_fidelity_sha256"], "fidelity pin")
    require(sha256(root / "RELEASE.json") == release["release_sha256"], "release pin")
    catalog = read_json(root / "SHA256SUMS.json")
    require(sha256(root / "SHA256SUMS.json") == release["checksums_sha256"], "catalog pin")
    membership = root / "added_subset_membership.json"
    require(sha256(membership) == catalog[membership.name], "membership pin")
    members = read_json(membership)
    manifest_path = delta / "image_manifest.json"
    annotations_path = delta / "annotations_coco.json"
    require(sha256(manifest_path) == fidelity["delta_manifest_sha256"], "added manifest pin")
    require(sha256(annotations_path) == fidelity["delta_annotation_sha256"], "added annotations pin")
    image_ids = set(members["image_ids"])
    require({im["image_id"] for im in read_json(manifest_path)["images"]} == image_ids,
            "unexpected consumer view membership")
    started = time.monotonic()
    records, counts = load_training_package(delta)
    require({r["image_id"] for r in records} == image_ids, "wrong consumer record set")
    require(counts["images"] == len(image_ids) == release["added_images"], "image count")
    require(counts["instances"] + counts["crowd_skipped"] == release["added_instances"], "mask count")
    write_json(args.output, {
        "status": "passed", "view": str(delta), "release_sha256": release["release_sha256"],
        "manifest_sha256": fidelity["delta_manifest_sha256"],
        "annotations_sha256": fidelity["delta_annotation_sha256"],
        "loader_sha256": sha256(inspect.getfile(load_training_package)),
        "elapsed_seconds": round(time.monotonic() - started, 2), "loader": counts,
    })


if __name__ == "__main__":
    main()
