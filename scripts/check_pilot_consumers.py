"""CPU consumer checks: COCO loading, source mask fidelity and region crop alignment."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from pycocotools.coco import COCO

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fashion_grounding.data.schema import decode_mask, read_json, require, write_json

parser = argparse.ArgumentParser()
parser.add_argument("package", type=Path)
args = parser.parse_args()
root = args.package
coco = COCO()
coco.dataset = read_json(root / "annotations_coco.json")
coco.createIndex()
source = COCO()
source.dataset = read_json(root / "source_subset.json")
source.createIndex()
package = read_json(root / "candidates_gt.json")
source_config = read_json(root / "mapping_config.json")
mapping = {r["source_id"]: r for r in source_config["fashionpedia"]}
expected = {a["id"] for a in source.anns.values() if mapping[a["category_id"]]["action"] == "map"}
actual = [a["source_annotation_id"] for a in coco.anns.values()]
require(set(actual) == expected and len(actual) == len(set(actual)), "whole source annotation coverage mismatch")
for aid, ann in coco.anns.items():
    raw = source.anns[ann["source_annotation_id"]]
    require(np.array_equal(coco.annToMask(ann), source.annToMask(raw)), "source mask changed")
    require(ann["category_id"] == mapping[raw["category_id"]]["project_category_id"], "wrong mapped category")
crop_count, hash_images = 0, []
for row in package["images"]:
    im = coco.imgs[row["image_id"]]
    with Image.open(root / im["file_name"]) as opened:
        image = opened.convert("RGB")
    for candidate in row["candidates"]:
        box = candidate["bbox_xyxy"]
        crop = image.crop(box)
        require(crop.size == (box[2]-box[0], box[3]-box[1]) and crop.width > 0 and crop.height > 0, "bad crop")
        crop_count += 1
    gray = np.asarray(image.convert("L").resize((32, 32)), dtype=float)
    n = np.arange(32)
    basis = np.cos(np.pi * (2*n[None, :]+1) * n[:, None] / 64)
    low = (basis @ gray @ basis.T)[:8, :8].flatten()[1:]
    hash_images.append((row["image_id"], low > np.median(low)))
near_pairs = []
for i, (iid, bits) in enumerate(hash_images):
    for other, other_bits in hash_images[i+1:]:
        distance = int(np.count_nonzero(bits != other_bits))
        if distance <= 6:
            near_pairs.append({"image_ids": [iid, other], "phash_distance": distance})
report = {"status": "passed", "coco_images_loaded": len(coco.imgs), "source_masks_pixel_identical": len(coco.anns), "candidate_crops_checked": crop_count, "phash_method": "32x32 grayscale DCT, top-left 8x8 excluding DC, median, Hamming <=6", "near_duplicate_pairs_within_pilot": near_pairs, "limitations": ["CPU file/crop checks only; no Detectron2/SigLIP model run or teammate acceptance", "pHash is a candidate finder, not proof of no duplicates; no cross-split/full-dataset duplicate audit", "This check does not perform human review; consult queries_accepted.jsonl and review_release.json when present"]}
write_json(root / "consumer_check_report.json", report)
print(json.dumps(report, indent=2))
